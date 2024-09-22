import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image as PIL_Image
from patch_ablation_utils import PatchMaster
from tqdm import tqdm
from pathlib import Path

import torchvision.transforms.functional as F
from PIL import Image




class CenterCropLongEdge():
    """
    https://github.com/facebookresearch/DIG-In/blob/main/utils.py#L21
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))
    
    def __repr__(self) -> str:
        return self.__class__.__name__
    
class PadToSquare():
    def __call__(self, img):
        max_side = max(img.size)
        delta_w = max_side - img.size[0]
        delta_h = max_side - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return transforms.functional.pad(img, padding, fill=0)
    
    def __repr__(self) -> str:
        return self.__class__.__name__

transforms_list = transforms.Compose(
        [
            CenterCropLongEdge(),
            transforms.Resize(224),
            transforms.ToTensor()]
)

transforms_list_with_padding = transforms.Compose(
        [
            PadToSquare(),
            CenterCropLongEdge(),
            transforms.Resize(224),
            transforms.ToTensor()]
)

pm = PatchMaster("vit")

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def mask_image(image, patch_mask, filler_val= 0, n_patches=14, patch_size=16):
    c,h,w = image.shape
    img = torch.ones(image.shape)*filler_val
    
    for i in patch_mask:
        if i>0: 
            x = (i-1)//n_patches
            y = (i-1)%n_patches
            img[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size] = image[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size]

    return img

def get_attention_masks(patches, device="cuda"):
    # patches: batch x (n_patch*n_patch+1)
    # return torch.hstack((torch.ones(patches.shape[0],1).to(device),torch.any(patches!=0, dim=-1)))
    return torch.any(patches!=0, dim=-1)


def mask_image(image, patch_mask, filler_val= 0, n_patches=14, patch_size=16):
    c,h,w = image.shape
    img = torch.ones(image.shape)*filler_val
    
    for i in patch_mask:
        if i>0: 
            x = (i-1)//n_patches
            y = (i-1)%n_patches
            img[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size] = image[:, x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size]

    return img

def get_masked_image(img_path, n_patches, patch_size, filler_val=0, with_padding=False):
    image = PIL_Image.open(img_path)
    if with_padding:
        image = transforms_list_with_padding(image)
    else:
        image = transforms_list(image) # resizing
    patches = patchify(image.unsqueeze(0), n_patches)
    mask_mod = get_attention_masks(patches, "cpu")
    ordering = torch.hstack(((mask_mod[0]==0).nonzero()[:,0], mask_mod[0].nonzero()[:,0]))
    ordering = ordering.unsqueeze(0).repeat(3,1)
    nums_to_exclude = (mask_mod[0]==0).nonzero()[:,0].shape[0]
    patch_mask = pm.get_patch_masks(ordering, nums_to_exclude)
    return mask_image(image, patch_mask[0], filler_val, n_patches, patch_size)

def border_and_alpha(img, patch_mask, patch_size=16,n_patches=14, fill_alpha=True, alpha_filler = 125, border_width=2):
    pil_img = F.to_pil_image(img)
    ni0 = np.array(pil_img)
    if fill_alpha:
        ni = np.ones((224,224,4), dtype='uint8')*255
        ni[:,:,:3] = ni0[:,:,:]
    else:
        ni = ni0
    mask = np.ones((224,224))
    for i in patch_mask[0]:
        if i>0: 
            x = (i-1)//n_patches
            y = (i-1)%n_patches
            mask[x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size] = 0.0
            if fill_alpha:
                ni[x*patch_size: (x+1)*patch_size, y*patch_size: (y+1)*patch_size, 3] = alpha_filler
    v_mask = np.zeros(mask.shape)
    h_mask = np.zeros(mask.shape)

    for i in range(1, mask.shape[0]):
        v_mask[i,:] = np.abs(mask[i-1,:]-mask[i,:])

    for j in range(1, mask.shape[1]):
        h_mask[:,j] = np.abs(mask[:, j-1]- mask[:, j])

    border = (v_mask +h_mask >0.5)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if border[i][j]==True:
                ni[i,j,0] = 255
                ni[i,j,1] = 0
                ni[i,j,2] = 0
                for k in range(border_width):
                    ni[max(0, i - k):min(224, i + k + 1), max(0, j - k):min(224, j + k + 1), 0] = 255
                    ni[max(0, i - k):min(224, i + k + 1), max(0, j - k):min(224, j + k + 1), 1] = 0
                    ni[max(0, i - k):min(224, i + k + 1), max(0, j - k):min(224, j + k + 1), 2] = 0
                    
                if fill_alpha:
                    ni[i,j,3] = 255
    return Image.fromarray(ni)