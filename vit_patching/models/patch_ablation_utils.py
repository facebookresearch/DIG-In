import sys
import torch
import numpy as np
import seaborn as sns
import pandas as pd
# import captum
from tqdm import tqdm
import time
import torchvision

def reduce_mask(mask): # mask is 1 if we are masking, 0 otherwise. return the patch orders
    # mask is B x 224 x 224
    to_keep = (torch.nn.MaxPool2d(16)(mask) == 0).squeeze(1) # B x 1 x 14 x 14
    max_num = (to_keep.int()).sum(dim=1).sum(dim=1).max()
    patches_to_keep = (torch.ones(mask.shape[0], max_num+1) * -1).cuda()
    patches_to_keep[:, 0] = 0
    for b in range(mask.shape[0]):
        coords, = torch.where(to_keep[b].flatten())
        coords = coords + 1
        patches_to_keep[b, 1:len(coords)+1] = coords
    return patches_to_keep.long()

class BasePatchMaster:
    def __init__(self, arch, img_size, total_patches):
        self.arch = arch
        self.img_size = img_size
        self.total_patches = total_patches
    
    def batch_mask_image(self, images_, patches_to_mask, filler_value, superpixel_extension=None):
        raise NotImplementedError
    
    def mask_images(self, patch_orders, nums_to_exclude, images, filler_value=(0,0,0), superpixel_extension=None):
        return self.batch_mask_image(
            images, 
            patches_to_mask=patch_orders[:, :nums_to_exclude], 
            filler_value=filler_value, 
            superpixel_extension=superpixel_extension)
    
    def slow_mask_orders(self, patch_orders, nums_to_exclude, images, superpixel_extension=None):
        masks = torch.zeros_like(images)
        masks = self.batch_mask_image(
            masks, 
            patches_to_mask=patch_orders[:, :nums_to_exclude], 
            filler_value=(1, 1, 1), 
            superpixel_extension=superpixel_extension)
        masks = masks[:, 0]
        patches_to_keep = reduce_mask(masks)
        return patches_to_keep
    
    def mask_image(self, image, patches_to_mask, filler_value=(0,0,0), superpixel_extension=None):
        if superpixel_extension is not None:
            superpixel_extension = superpixel_extension.unsqueeze(0)
        masked = self.batch_mask_image(
            images=image.unsqueeze(0), 
            patches_to_mask=patches_to_mask, 
            filler_value=filler_value,
            superpixel_extension=superpixel_extension,
        )
        return masked.squeeze(0)

    def generate_random_patch_orders(self, B):
        patch_orders = []
        for j in range(B):
            order = np.arange(self.total_patches)
            np.random.shuffle(order)
            patch_orders.append(order)
        patch_orders = np.stack(patch_orders)
        return torch.tensor(patch_orders)
    

class SuperPixelPatchMaster(BasePatchMaster):
    def batch_mask_image(self, images_, patches_to_mask, filler_value, superpixel_extension=None):
        # superpixel_extension is N x Total_Patches x H x W
        assert superpixel_extension is not None
        images = images_.clone()
        for i in range(images.shape[0]):
            img = images[i]
            superpixel_masks = superpixel_extension[i, patches_to_mask[i]]
            summed_mask = superpixel_masks.sum(dim=0) > 0
            for j in range(3):
                if filler_value[j] == -1:
                    val = np.random.random()
                elif filler_value[j] == -2:
                    sz = img[j, summed_mask].shape
                    val = torch.rand(size=sz)
                else:
                    val = filler_value[j]
                img[j, summed_mask] = val
        return images
    
    def generate_random_patch_orders(self, superpixels, num_features):
        patch_orders = []
        B = superpixels.shape[0]
        for j in range(B):
            total_patches = superpixels[j].max() + 1
            order = np.arange(total_patches)
            np.random.shuffle(order)
            patch_orders.append(order[:num_features])
        patch_orders = np.stack(patch_orders)
        return torch.tensor(patch_orders)
    
    def get_superpixel_patch_ordering(self, S, num_features, superpixels, reverse, use_mean=True):
        '''
        Given saliency map, chunk the saliency map and then order the patches
        by their mean value.
        input: N x C x H x W
        Returns: the flattened patch means and their order
        '''
        B = superpixels.shape[0]
        patch_orders = []
        for b in range(B):
            superpixel = superpixels[b]
            saliency = S[b]
            num_patches = superpixel.max() + 1
            saliency_vals = torch.stack([saliency[:, superpixel == i].mean() for i in range(num_patches)])
            if reverse:
                saliency_indices = saliency_vals.argsort(descending=False)[:num_features]
            else:
                saliency_indices = saliency_vals.argsort(descending=True)[:num_features]
            patch_orders.append(saliency_indices)
        return torch.stack(patch_orders)
    
    def batch_single_mask_image(self, images_, patches_to_mask, filler_value, superpixel_extension=None):
        # superpixel_extension is N x Total_Patches x H x W
        # patches_to_mask is exactly 1 vector of patches_to_mask
        assert superpixel_extension is not None
        images = images_.clone().permute(0, 2, 3, 1)
        superpixel_masks = superpixel_extension[:, patches_to_mask]
        summed_mask = superpixel_masks.sum(dim=1) > 0
        images[summed_mask] = torch.tensor(filler_value).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)
        return images

    def slow_mask_orders_single_image(self, images_, patches_to_mask, filler_value, superpixel_extension=None):
        masks = torch.zeros_like(images_)
        masks = self.batch_single_mask_image(
            images_=masks, 
            patches_to_mask=patches_to_mask, 
            filler_value=(1, 1, 1), 
            superpixel_extension=superpixel_extension)
        masks = masks[:, 0]
        patches_to_keep = reduce_mask(masks)
        return patches_to_keep
          
class PatchMaster(BasePatchMaster):
    def __init__(self, arch, img_size=224, patch_size=16):
        self.num_patches = img_size//patch_size        
        self.patch_size = patch_size
        super().__init__(arch=arch, img_size=img_size,total_patches=self.num_patches ** 2)
        self.patch_indexer = self.get_patch_indexer()
        
    def get_patch_indexer(self):
        index_arr = torch.arange(self.img_size*self.img_size).reshape(self.img_size, self.img_size)
        chunk1 = torch.stack(torch.chunk(index_arr, self.num_patches, dim=0), dim=0)
        chunk2 = torch.stack(torch.chunk(chunk1, self.num_patches, dim=-1), dim=1)
        chunk3 = chunk2.flatten(2,3)
        chunk4 = chunk3.flatten(0,1)
        return chunk4 # [196, 256]
        
    def get_patch_ordering(self, S, use_mean=True):
        '''
        Given saliency map, chunk the saliency map into patches and then order the patches
        by their mean value.
        input: N x C x H x W
        Returns: the flattened patch means and their order
        '''
        chunk1 = torch.stack(torch.chunk(S, self.num_patches, 2), 1)
        chunk2 = torch.stack(torch.chunk(chunk1, self.num_patches, -1), 2)
        chunk3 = chunk2.flatten(3, 5)
        if use_mean:
            chunk4 = chunk3.mean(dim=-1)
        else:
            chunk4, _ = chunk3.max(dim=-1)
        chunk5 = chunk4.flatten(1,2)
        order = chunk5.argsort(descending=True, dim=-1)
        return chunk5, order

    def get_patch_masks(self, patch_orders, nums_to_exclude):
        '''
        Given patch orderings, return patch masks which exclude the first nums_to_exclude patches
        '''
        all_patch_masks = []
        for order in patch_orders:
            exclude_set = order[:nums_to_exclude]
            patch_mask = [r+1 for r in np.arange(self.total_patches) if r not in exclude_set]
            patch_mask = [0] + patch_mask
            all_patch_masks.append(np.array(patch_mask))
        all_patch_masks = np.stack(all_patch_masks)
        return torch.tensor(all_patch_masks).long()
    

    
    def batch_mask_image(self, images_, patches_to_mask, filler_value, superpixel_extension=None):
        images = images_.clone()
        patch_indexing = self.patch_indexer[patches_to_mask].view(images.shape[0], -1)
        if -3 in filler_value:
            blurred_value = torchvision.transforms.GaussianBlur(kernel_size=21, sigma=10)(images_)
        for j in range(images.shape[0]):
            for i in range(3):
                if filler_value[i] == -1:
                    val = np.random.random()
                elif filler_value[i] == -2:
                    sz = images.view(*images.shape[:2], -1)[j, i][patch_indexing[j]].shape
                    val = torch.rand(size=sz)
                elif filler_value[i] == -3:
                    val = blurred_value.view(*images.shape[:2], -1)[j, i][patch_indexing[j]]
                else:
                    assert filler_value[i] >= 0
                    val = filler_value[i]
                images.view(*images.shape[:2], -1)[j, i][patch_indexing[j]] = val
        return images
