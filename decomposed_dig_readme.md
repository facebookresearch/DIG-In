
# Decomposed evaluations of geographic disparities in text-to-image models

This library contains code for measuring disparities in text-to-image generative models as introduced in [Decomposed evaluations of geographic disparities in text-to-image models](https://arxiv.org/abs/2406.11988). 
This library supports evaluating disparities in generated image quality, diversity, and consistency between geographic regions, using [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/)

Learn more about the development of these Indicators and how they can be used for auditing text-to-image generative models in the original DIG-In [paper](https://arxiv.org/abs/2308.06198). 

## Details about included files

In particular, users are instructed to complete the following steps:

### [1] Generate images
Generate images corresponding to the prompts in the following csvs. Each csv should correspond to a single folder with one image per row in the csv. The image should follow the  naming scheme `[prompt]__[imgid].png` as defined below: 
* `geode_prompts_regions_fulldataset.csv`
* `geode_prompts_countries_fulldataset.csv`


### [2] Extract features using patch ViT
These scripts require pointers to a prompt csv and folder of generated images and yield a pickle file containing image features for each generated image. 
This file matches the structure of the prompt csv.
* patchViT: `get_features_patch_vit.py`

```
'df_path': prompts csv
'img_dir': folder containing the images
'do_obj_seg': whether to extract object features or background features
'use_compliment_for_BG': true means, background patches are the ones that are not used in object segmentation
'save_path': features save path
'model_name': which model to use; `vit_base_patch16_224` is the default
```

Sample command to extract patch ViT features for Obj segmented part of image:

```
python get_features_patch_vit.py --df_path "/home/asureddy_umass_edu/DIG-In/data/geode_prompts_regions_only_balanceddataset_processed.csv" --img_dir "/work/pi_dhruveshpate_umass_edu/project_21/data/generated-data/LDM-1-5/geode_prompts_regions_balanceddataset" --do_obj_seg --save_path "data_sample2.pkl" --use_compliment_for_BG
```

Sample command to extract patch ViT features for BG segmented part of image:

```
python get_features_patch_vit.py --df_path "/home/asureddy_umass_edu/DIG-In/data/geode_prompts_regions_only_balanceddataset_processed.csv" --img_dir "/work/pi_dhruveshpate_umass_edu/project_21/data/generated-data/LDM-1-5/geode_prompts_regions_balanceddataset" --save_path "data_sample2.pkl" --use_compliment_for_BG
```

> Note, if you use the path `{features_folder}/{which_dataset}_prompts_[regions/countries]_fulldataset_{which_model}_{which_features}_[obj/bg]_deduped.pkl` for saving your features then you should be able to go to the next step without updating df paths.

### [3] Compute Indicators
These scripts require a pointer to the pickle of image features created in the previous step and yield a folder with csvs containing some subset of precision, recall, coverage, and density . Note that depending on how you saved the features in Step \#2, you may need to update the paths corresponding to the features. The script for calculating metrics, inc. balancing reference datasets, can be found in `compute_metrics_decomposed_dig.py`.

This script can be run with the following arguments to calculate respective Indicators:
1. Region Indicator: 
```
k_PR=3 
which_metric=pergroup_pr 
ref_features_path = path of referrence image features
gen_features_path = path of generated image features
save_metrics_path = compute metrics save path
remove_unmasked_geode = if true, removes reference images that are not segmented by LangSAM
is_bg_image = whether it is bg segmented DIG computation or obj segmented DIG computation
```
2. Region-Object Indicator: 
```
k_PR=3 
ref_features_path = path of referrence image features
gen_features_path = path of generated image features
save_metrics_path = compute metrics save path
remove_unmasked_geode = if true, removes reference images that are not segmented by LangSAM
is_bg_image = whether it is bg segmented DIG computation or obj segmented DIG computation
```


## License


## Citation

If you use the Decomposed DIG Indicators or if the work is useful in your research, please give us a star and cite: 

```
@misc{sureddy2024decomposedevaluationsgeographicdisparities,
      title={Decomposed evaluations of geographic disparities in text-to-image models}, 
      author={Abhishek Sureddy and Dishant Padalia and Nandhinee Periyakaruppa and Oindrila Saha and Adina Williams and Adriana Romero-Soriano and Megan Richards and Polina Kirichenko and Melissa Hall},
      year={2024},
      eprint={2406.11988},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.11988}, 
}
```