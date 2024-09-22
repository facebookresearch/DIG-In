import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import ViT_B_16_Weights
import torchvision.transforms as transforms
from lang_sam import LangSAM
from PIL import Image as PIL_Image
from vit_patching.models.patch_ablation_utils import PatchMaster
from vit_patching.models import vision_transformer
## The ViT patch code was referred from https://github.com/MadryLab/missingness/tree/master/src 
from tqdm import tqdm
from pathlib import Path

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("-df_path", "--df_path", type=str)
parser.add_argument("-img_dir", "--img_dir",type=str)
parser.add_argument("--do_obj_seg", action="store_true")
parser.add_argument("--use_compliment_for_BG", action="store_true")
parser.add_argument("--use_blackened", action="store_true")
parser.add_argument("-save_path", "--save_path", type=str)
parser.add_argument("-model_name", "--model_name", type=str, default="vit_base_patch16_224", help="Model name, e.g., 'vit_base_patch16_224' or 'vit_large_patch32_384'")
args = parser.parse_args()

class CenterCropLongEdge():
    """
    https://github.com/facebookresearch/DIG-In/blob/main/utils.py#L21
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))
    
    def __repr__(self) -> str:
        return self.__class__.__name__

TRANSFORM_NORMALIZE = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
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


def relevant_image_and_patch_mask(image_pil, numpy_mask, n_patches, preprocessor1, do_obj_seg, use_compliment_for_BG, use_blackened):
    numpy_image = np.array(image_pil)
    if use_compliment_for_BG:
        # BG seg patches are compliment of Obj seg patches
        numpy_image[numpy_mask == False] = [0, 0, 0]  
        obj_seg_image = PIL_Image.fromarray(numpy_image) # object segmented image
        obj_seg_image = preprocessor1(obj_seg_image) # converting to 224x224

        patches = patchify(obj_seg_image.unsqueeze(0), n_patches)
        mask_mod = get_attention_masks(patches, "cpu")

        if do_obj_seg:
            ordering = torch.hstack(((mask_mod[0]==0).nonzero()[:,0], mask_mod[0].nonzero()[:,0]))
            nums_to_exclude = (mask_mod[0]==0).nonzero()[:,0].shape[0]
        else:
            ordering = torch.hstack((mask_mod[0].nonzero()[:,0], (mask_mod[0]==0).nonzero()[:,0]))
            nums_to_exclude = mask_mod[0].nonzero()[:,0].shape[0]

        ordering = ordering.unsqueeze(0).repeat(3,1)
        patch_mask = pm.get_patch_masks(ordering, nums_to_exclude) # extracting the relevant patches on segmented part

        patch_mask = patch_mask[0,:].unsqueeze(0)

        if use_blackened:
            # setting: (blackened Obj, all obj patches) , (blackened BG, all non obj patches)
            numpy_image = np.array(image_pil)
            numpy_image[numpy_mask != do_obj_seg] = [0,0,0]
            seg_image = PIL_Image.fromarray(numpy_image)
            seg_image = preprocessor1(seg_image)
            return seg_image, patch_mask
        else:
            # setting: (orig Obj, all obj patches) , (orig BG, all non obj patches)
            # processing original image
            # blackened patch parts are replaced by original image parts
            image_pil = preprocessor1(image_pil)
            return image_pil, patch_mask
    else:
        # take all patches where the seg image is present
        numpy_image[numpy_mask != do_obj_seg] = [0,0,0]
        seg_image = PIL_Image.fromarray(numpy_image)
        seg_image = preprocessor1(seg_image)

        patches = patchify(seg_image.unsqueeze(0), n_patches)
        mask_mod = get_attention_masks(patches, "cpu")
        ordering = torch.hstack(((mask_mod[0]==0).nonzero()[:,0], mask_mod[0].nonzero()[:,0]))
        nums_to_exclude = (mask_mod[0]==0).nonzero()[:,0].shape[0]

        ordering = ordering.unsqueeze(0).repeat(3,1)
        patch_mask = pm.get_patch_masks(ordering, nums_to_exclude) # extracting the relevant patches on segmented part
        patch_mask = patch_mask[0,:].unsqueeze(0)

        if use_blackened:
            # setting: (blackened Obj, all obj patches), (blackened BG, all BG patches)
            return seg_image, patch_mask
        else:
            # setting: (orig Obj, all obj patches), (Orig BG, all BG patches)
            image_pil = preprocessor1(image_pil)
            return image_pil, patch_mask

def generate_features_new(df_path, img_dir, do_obj_seg, save_path, model_name, use_blackened, use_compliment_for_BG):
    print(f"{df_path, img_dir, do_obj_seg, save_path, model_name, use_blackened, use_compliment_for_BG}")
    if model_name == "vit_base_patch16_224":
        transformer = vision_transformer.vit_base_patch16_224(pretrained=True).to("cuda")
        img_size, patch_size = 224, 16
    elif model_name == "vit_base_patch32_384":
        transformer = vision_transformer.vit_base_patch32_384(pretrained=True).to("cuda")
        img_size, patch_size = 384, 32
    elif model_name == "vit_large_patch16_224":
        transformer = vision_transformer.vit_large_patch16_224(pretrained=True).to("cuda")
        img_size, patch_size = 224, 16
    elif model_name == "vit_large_patch32_384":
        transformer = vision_transformer.vit_large_patch32_384(pretrained=True).to("cuda")
        img_size, patch_size = 384, 32
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    out_size = 768
    # langsam model
    model = LangSAM()
    
    n_patches = img_size // patch_size
    
    transforms_list = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(img_size),  # Use dynamic img_size based on the model
        transforms.ToTensor()
    ])
    
    preprocessor1 = transforms_list
    preprocessor2 = TRANSFORM_NORMALIZE
    # reading the data
    df = pd.read_csv(df_path)
    df = df.head()

    features = []
    is_good_features = []
    masked = []
    patches_considered = []
    for _,row in tqdm(df.iterrows()):
        is_masked = 0
        img_path = Path(f"{img_dir}/{row['file_path']}")
        obj = row['object']
        if img_path.is_file():
            try: 
                image_pil = PIL_Image.open(img_path) # original image
                masks, boxes, phrases, logits = model.predict(image_pil, obj)
                if len(masks)==0:
                    print(f"couldn't mask: {img_path}")
                    features.append([0.0]*out_size)
                    is_good_features.append(1)
                    masked.append(is_masked)
                    patches_considered.append(-1)
                    continue
                is_masked = 1
                with torch.no_grad():
                    merged_tensor = torch.any(masks, dim=0, keepdim=True)
                    numpy_mask = merged_tensor.squeeze().numpy()
                    res_img, patch_mask = relevant_image_and_patch_mask(image_pil, numpy_mask, n_patches, preprocessor1, do_obj_seg, use_compliment_for_BG, use_blackened)
                    
                    patch_mask = patch_mask.cuda()
                    res_img = preprocessor2(res_img)
                    res_img = res_img.cuda()
                    feature = transformer.forward_features(res_img.unsqueeze(0), patch_mask)[0].cpu().detach().numpy()
                    features.append(
                        feature
                    )
                    patches_considered.append(patch_mask.shape[-1])
                    is_good_features.append(1)
                    masked.append(is_masked)
                    # print(feature)
                    # print("processed")
                

            except Exception as e:
                print(e)
                features.append([])
                is_good_features.append(0)
                masked.append(is_masked)
                patches_considered.append(-1)
        else:
            print(f"file not found: {img_path}")
            features.append([])
            is_good_features.append(0)
            masked.append(is_masked)
            patches_considered.append(-1)


    df["features"] = features
    df["is_good"] = is_good_features
    df["masked"] = masked
    df["patches_considered"] = patches_considered
    print(f"masks found for {sum(masked)} out of {len(masked)}")
    print(f"feature extraction done for {sum(is_good_features)} out of {len(is_good_features)}")
    df = df[df["is_good"]==1]

    df.to_pickle(save_path)
    

# data = "/home/asureddy_umass_edu/DIG-In/data/geode_prompts_regions_only_balanceddataset_processed.csv"
# img_dir = "/work/pi_dhruveshpate_umass_edu/project_21/data/generated-data/LDM-1-5/geode_prompts_regions_balanceddataset"

df_path = args.df_path
img_dir = args.img_dir
do_obj_seg = args.do_obj_seg
save_path = args.save_path
model_name = args.model_name
use_compliment_for_BG = args.use_compliment_for_BG
use_blackened = args.use_blackened

generate_features_new(df_path, img_dir, do_obj_seg, save_path, model_name, False, True)