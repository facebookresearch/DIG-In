import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import manifold_metrics as mm

import configargparse
import pickle5 as pickle


## Load data
parser = configargparse.ArgumentParser()
parser.add_argument("-k_PR", "--k_PR", type=int)
parser.add_argument("-which_metric", "--which_metric", type=str)
parser.add_argument("-ref_features_path", "--ref_features_path", type=str)
parser.add_argument("-gen_features_path", "--gen_features_path", type=str)
parser.add_argument("-save_metrics_path", "--save_metrics_path", type=str)
# parser.add_argument("--balance", action="store_true")
parser.add_argument("--remove_unmasked_geode", action="store_true")
parser.add_argument("--is_full_image", action="store_true")
parser.add_argument("--is_bg_image", action="store_true")
args = parser.parse_args()

ref_features_path = args.ref_features_path
gen_features_path = args.gen_features_path
# apply_filt = args.balance
k_PR = args.k_PR
which_metric = args.which_metric

save_metrics_path = args.save_metrics_path



# balancing the real geode dataset
balancing = True
remove_unmasked_geode = args.remove_unmasked_geode
is_full_image = args.is_full_image
is_bg_image = args.is_bg_image


# reference data
# dataset_df = pd.read_pickle(ref_features_path)
try:
    with open(ref_features_path, 'rb') as f:
        dataset_df = pickle.load(f)
except:
    print("using pandas to read real features")
    dataset_df = pd.read_pickle(ref_features_path)
    
dataset_df["features"] = [list(i) for i in dataset_df["features"]]
dataset_df["id"] = dataset_df["file_path"]
dataset_df["str_features"] = [
    " ".join([str(i) for i in k]) for k in dataset_df["features"]
]

dataset_df['r'] = dataset_df['region']

# removing images having patches <= 25
if not is_full_image and is_bg_image:
    print("not full image, so removing images with patches < 25")
    # this 25 can be a hyper parameter
    dataset_df = dataset_df[dataset_df["patches_considered"]>25]



# object_file = "objects_180.txt"
# obj_count_per_reg = 175
# changing for the sake of BG seg
obj_count_per_reg = 172
# to remove 3 objects from the 27 objects where the no masks > 100
# 21 objects
object_file = "seg_low_patches_removed_objects.pkl"

# balancing real GeoDE dataset
if balancing:
    # using 175 images per object region
    # 175 is the minimum number of real images in a object-region setting
    # 1. remove few classes from real hold which are not segmented properly
    # 2. remove no masks from real hold (automatically done in get_df_deduped)
    # Address images with identical features
    if remove_unmasked_geode:
        # this flag should be true for real geode features to take exact same images set as we took for 
        # reference segmented dataset
        dataset_df = dataset_df[dataset_df["masked"]==1]

    deduped_dataset_df = utils.get_df_deduped(dataset_df, keep_one=False)
    # 3. balancing the real hold
    deduped_dataset_df = utils.sample_df(deduped_dataset_df, object_file, 42, obj_count_per_reg)
object_key = "object"

print(f"Data: {deduped_dataset_df.shape}")


# generated dataset
# model_df = pd.read_pickle(gen_features_path)
try:
    with open(gen_features_path, 'rb') as f:
        model_df = pickle.load(f)
except:
    print("using pandas to load pickle file")
    model_df = pd.read_pickle(gen_features_path)
    
model_df["r"] = (
        model_df["region"]
        .replace("the Americas", "Americas")
        .replace("East Asia", "EastAsia")
        .replace("Southeast Asia", "SouthEastAsia")
        .replace("West Asia", "WestAsia")
    )

model_df["object"] = [i.replace(" ", "_") for i in model_df["object"]]
features_key = "features"

if model_df is not None:
    model_df[features_key] = [list(i) for i in model_df[features_key]]
    print(f"Fake: {model_df.shape}")

if balancing:
    # 4. to balance generated images to same count as real images
    model_df = utils.sample_df(model_df, object_file, 42, obj_count_per_reg)

if is_bg_image:
    # low BG patch images => masking not done for BG
    print("In generated images, images with number of patches < 25, considered as un-masked")
    model_df["masked"] =  model_df["masked"]*(model_df["patches_considered"]>25)

# Prepare real and fake dataframes
real_hold = deduped_dataset_df
fake_hold_r = model_df[model_df.r.isin(list(set(real_hold.r)))]


print(
        f"Real: {real_hold.shape}, fake R:{fake_hold_r.shape}"
    )

# real_features = np.array(real_hold["features"].values.tolist())
# fake_features = np.array(fake_hold[features_key].values.tolist())
# fake_features_r = np.array(fake_hold_r[features_key].values.tolist())


# PC functions
# only precision and coverage functions
def get_grouped_pc_and_objects_segmented(real_hold, fake_hold, nearest_k, is_full_image, filter_match=True, verbose=True ):
    # the steps were already done
    # 1. remove few classes from real hold
    # 2. remove no masks from real hold
    # 3. balancing the real hold
    # 4. balancing the fake hold
    # @Polina: To double check my understanding of this approach of swapping out generated images that were not masked with all-zeros data: 
    # is it correct that for the precision score it would be equivalent to setting the indicator for these images to 0 in the formula on the screenshot?
    # And for coverage score that would mean those images are dropped from being considered to be lying in the hyperspheres of real data?
    # column name: masked, for un masked, value=0
    real_features = np.array(real_hold["features"].values.tolist())
    fake_features = np.array(fake_hold["features"].values.tolist())
    if not is_full_image:
        fake_features_w_masks = np.array(fake_hold[fake_hold["masked"]==1]["features"].values.tolist())
        fake_filter_w_masks = fake_hold[fake_hold["masked"]==1]["r"]
    else:
        fake_features_w_masks = fake_features
        fake_filter_w_masks = fake_hold["r"]

    print(f"total fake features: {len(fake_features)}")
    print(f"Total fake features with masking: {len(fake_features_w_masks)}")
    real_filter = real_hold["r"]
    fake_filter = fake_hold["r"]
    r_filters = []
    f_filters = []
    p = []
    c = []

    for r_filter in list(set(real_filter)):
        real_features_sub = real_features[real_filter == r_filter]
        # real_objects_sub = real_objects[real_filter == r_filter]
        print(
            f"Computing real manifold, real filter: {r_filter}; Num real: {real_features_sub.shape[0]}"
        )
        real_nearest_neighbour_distances = mm.compute_nearest_neighbour_distances(
            real_features_sub, nearest_k
        )

        for f_filter in list(set(fake_filter)):
            if filter_match and (r_filter != f_filter):
                continue
            # print(f_filter.shape)
            fake_features_sub = fake_features[fake_filter == f_filter]
            fake_features_w_masks_sub = fake_features_w_masks[fake_filter_w_masks == f_filter]

            if verbose:
                print(f"real filter: {r_filter}; fake filter: {f_filter}")
                print(f"Num real: {real_features_sub.shape[0]}, Num fake: {fake_features_sub.shape[0]}, Num fake w masks: {fake_features_w_masks_sub.shape[0]}")
                print(
                    "Num real: {} Num fake: {}".format(
                        real_features_sub.shape[0], fake_features_w_masks_sub.shape[0]
                    )
                )


            distance_real_fake = mm.compute_pairwise_distance(
                real_features_sub, fake_features_w_masks_sub
            )

            if verbose:
                print("Computing precision...")
            precision = (
                (
                    distance_real_fake
                    < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                ).any(axis=0)
            )
            precision = precision.sum() / fake_features_sub.shape[0]

            if verbose:
                print("Computing coverage...")
            coverage = distance_real_fake.min(axis=1) < real_nearest_neighbour_distances

            r_filters.append(r_filter)
            f_filters.append(f_filter)
            p.append(precision)
            c.append(coverage.mean())

    return pd.DataFrame(
        zip(r_filters, f_filters, p, c),
        columns=[
            "real_filter",
            "fake_filter",
            "precision",
            "coverage",
        ],
    )

def get_grouped_pc_and_objects_segmented_perobject(real_hold, fake_hold, nearest_k, is_full_image, filter_match=True, verbose=True ):
    # the steps were already done
    # 1. remove few classes from real hold
    # 2. remove no masks from real hold
    # 3. balancing the real hold
    # 4. balancing the fake hold
    # @Polina: To double check my understanding of this approach of swapping out generated images that were not masked with all-zeros data: 
    # is it correct that for the precision score it would be equivalent to setting the indicator for these images to 0 in the formula on the screenshot?
    # And for coverage score that would mean those images are dropped from being considered to be lying in the hyperspheres of real data?
    # column name: masked, for un masked, value=0
    real_features = np.array(real_hold["features"].values.tolist())
    fake_features = np.array(fake_hold["features"].values.tolist())
    if not is_full_image:
        fake_features_w_masks = np.array(fake_hold[fake_hold["masked"]==1]["features"].values.tolist())
        fake_filter_w_masks = fake_hold[fake_hold["masked"]==1]["r"]
    else:
        fake_features_w_masks = fake_features
        fake_filter_w_masks = fake_hold["r"]

    print(f"total fake features: {len(fake_features)}")
    print(f"Total fake features with masking: {len(fake_features_w_masks)}")
    real_filter = real_hold["r"]
    fake_filter = fake_hold["r"]
    real_obj_filter = real_hold["object"]
    fake_obj_filter = fake_hold["object"]
    fake_obj_filter_w_masks = fake_hold[fake_hold["masked"]==1]["object"]
    r_filters = []
    f_filters = []
    o_filters = []
    p = []
    c = []

    for r_filter in list(set(real_filter)):
        for o_filter in list(set(real_obj_filter)):
            real_features_sub = real_features[(real_filter == r_filter)*(real_obj_filter==o_filter)]
            # real_objects_sub = real_objects[real_filter == r_filter]
            print(
                f"Computing real manifold, real filter: {r_filter}; Num real: {real_features_sub.shape[0]}; o_filter: {o_filter}"
            )
            real_nearest_neighbour_distances = mm.compute_nearest_neighbour_distances(
                real_features_sub, nearest_k
            )

            for f_filter in list(set(fake_filter)):
                if filter_match and (r_filter != f_filter):
                    continue
                # print(f_filter.shape)
                fake_features_sub = fake_features[(fake_filter == f_filter)*(fake_obj_filter==o_filter)]
                fake_features_w_masks_sub = fake_features_w_masks[(fake_filter_w_masks == f_filter)*(fake_obj_filter_w_masks==o_filter)]

                if verbose:
                    print(f"real filter: {r_filter}; fake filter: {f_filter}; o_filter: {o_filter}")
                    print(f"Num real: {real_features_sub.shape[0]}, Num fake: {fake_features_sub.shape[0]}, Num fake w masks: {fake_features_w_masks_sub.shape[0]}")
                    print(
                        "Num real: {} Num fake: {}".format(
                            real_features_sub.shape[0], fake_features_w_masks_sub.shape[0]
                        )
                    )


                distance_real_fake = mm.compute_pairwise_distance(
                    real_features_sub, fake_features_w_masks_sub
                )

                if verbose:
                    print("Computing precision...")
                precision = (
                    (
                        distance_real_fake
                        < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                    ).any(axis=0)
                )
                precision = precision.sum() / fake_features_sub.shape[0]

                if verbose:
                    print("Computing coverage...")
                coverage = distance_real_fake.min(axis=1) < real_nearest_neighbour_distances

                r_filters.append(r_filter)
                f_filters.append(f_filter)
                o_filters.append(o_filter)
                p.append(precision)
                c.append(coverage.mean())

    return pd.DataFrame(
        zip(r_filters, f_filters, o_filters, p, c),
        columns=[
            "real_filter",
            "fake_filter",
            "o_filter",
            "precision",
            "coverage",
        ],
    )

#######
######
## Precision/recall for {object} in {region} generations vs. real data
#######
#######
# all real vs all {object} in {region}
print(f">>>>>>> Computing {which_metric} for object in region prompt.")

if which_metric == "pergroup_pr":
    pr_df_region_g = get_grouped_pc_and_objects_segmented(real_hold, fake_hold_r, nearest_k=k_PR, is_full_image=is_full_image)
    pr_df_region_g.to_csv(
        save_metrics_path
    )

if which_metric == "perobj_pergroup_pr":
    pr_df_region_g_obj = get_grouped_pc_and_objects_segmented_perobject(real_hold, fake_hold_r, nearest_k=k_PR, is_full_image=is_full_image)
    pr_df_region_g_obj.to_csv(
        save_metrics_path
    )