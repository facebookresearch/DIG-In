# DIG In: Evaluating Disparities in Image Generation with Indicators for Geographic Diversity

This library contains code for measuring disparities in text-to-image generative models as introduced in [DIG In: Evaluating Disparities in Image Generations with Indicators for Geographic Diversity](https://arxiv.org/abs/2308.06198). 
This library supports evaluating disparities in generated image quality, diversity, and consistency between geographic regions, using [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/) and [DollarStreet](https://www.gapminder.org/dollar-street) as reference datasets. 

Learn more about the development of these Indicators and how they can be used for auditing text-to-image generative models in our [paper](https://arxiv.org/abs/2308.06198). 

## Details about included files
In this repository, users will find scripts to extract relevant features from generated images and calculate precision and coverage metrics using GeoDE or DollarStreet as reference datasets and CLIPScore metrics using GeoDE object prompts. 
In each of these scripts, there are `#TODOs` noting places where you should supply pointers to your image or feature paths. 

In particular, users are instructed to complete the following steps:

### [1] Generate images
Generate images corresponding to the prompts in the following csvs. Each csv should correspond to a single folder with one image per row in the csv. The image should follow the  naming scheme `[prompt]__[imgid].png` as defined below: 
* `geode_prompts_regions_fulldataset.csv`
* `geode_prompts_countries_fulldataset.csv`
* `ds_prompts_regions_fulldataset.csv`
* `ds_prompts_countries_fulldataset.csv`

To run measurements with GeoDE as the reference dataset, only the first two csvs are required. To run measurements with DollarStreet as the reference dataset, all four csvs are required. 

### [2] Extract features
These scripts require pointers to a prompt csv and folder of generated images and yield a pickle file containing image features for each generated image. 
This file matches the structure of the prompt csv.
* InceptionV3: `get_features_inceptionv3.py`
* CLIPScore: `get_features_clip_score.py`

> Note, if you use the path `{features_folder}/{which_dataset}_prompts_[regions/countries]_fulldataset_{which_model}_{which_features}_deduped.pkl` for saving your features then you should be able to go to the next step without updating df paths.

### [3] Compute Indicators
These scripts require a pointer to the pickle of image features created in the previous step and yield a folder with csvs containing some subset of precision, recall, coverage, and density (PRDC) and CLIPScore metrics. Note that depending on how you saved the features in Step \#2, you may need to update the paths corresponding to the features. The script for calculating metrics, inc. balancing reference datasets, can be found in `compute_metrics.py`.

This script can be run with the following arguments to calculate respective Indicators:
1. Region Indicator: 
```
which_model=[MODEL NAME]
k_PR=3 
which_metric=pergroup_pr 
which_features==inceptionv3 
which_dataset=[geode/ds]
```
2. Region-Object Indicator: 
```
which_model=[MODEL NAME]
k_PR=3 
which_metric=perobj_pergroup_pr 
which_features==inceptionv3 
which_dataset=[geode/ds]
```
3. Consistency Indicator:
```
which_model=[MODEL NAME]
k_PR=0
which_metric=clip_score
which_features==clip_score 
which_dataset=geode
```

## License
The majority of DIG In is licensed under CC-BY-NC, however portions of the project are available under separate license terms: PRDC metrics and InceptionV3 are licensed under the MIT license.

## Citation
If you use the DIG Indicators or if the work is useful in your research, please give us a star and cite:

```
@misc{hall2023dig,
      title={DIG In: Evaluating Disparities in Image Generations with Indicators for Geographic Diversity}, 
      author={Melissa Hall and Candace Ross and Adina Williams and Nicolas Carion and Michal Drozdzal and Adriana Romero Soriano},
      year={2023},
      eprint={2308.06198},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
