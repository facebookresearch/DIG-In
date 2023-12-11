# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# prdc
# Copyright (c) 2020-present NAVER Corp.
# MIT license

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import manifold_metrics as mm

import configargparse

# ## Load data
parser = configargparse.ArgumentParser()
parser.add_argument("-which_model", "--which_model", type=str)
parser.add_argument("-k_PR", "--k_PR", type=int)
parser.add_argument("-which_metric", "--which_metric", type=str)
parser.add_argument("-which_features", "--which_features", type=str)
parser.add_argument("-which_dataset", "--which_dataset", type=str)
parser.add_argument("--balance", action="store_true")
args = parser.parse_args()


which_model = args.which_model
which_dataset = args.which_dataset
apply_filt = args.balance
which_features = args.which_features
k_PR = args.k_PR
which_metric = args.which_metric

features_folder = ""  # TODO -- folder where features are saved

if apply_filt:
    ext = "_balanced"
else:
    ext = ""

if which_dataset == "ds":
    apply_filt = False  # no balancing for DS dataset otherwise esimated manifolds are too small number of datapoints

save_metrics_path = ""  # TODO -- where metrics should be saved
if not os.path.exists(save_metrics_path):
    # Create a new directory because it does not exist
    os.makedirs(save_metrics_path)
print(f"Saving metrics to {save_metrics_path}")

dataset_choices = ["geode", "ds"]
features_choices = ["inceptionv3"]

if which_dataset not in dataset_choices:
    raise ValueError(f"Dataset should be in {dataset_choices}")
if which_features not in features_choices:
    raise ValueError(f"Features should be in {features_choices}")

obj_map_ds = {
    "spices": "spices",
    "medication": "medicine",
    "cleaning equipment": "cleaning equipment",
    "tooth paste": "toothpaste toothpowder",
    "toys": "toy",
    "car": "car",
    "soap for hands and body": "hand soap",
    "plate of food": "plate of food",
    "front door": "front door",
    "toothbrush": "toothbrush",
    "cooking pots": "cooking pot",
    "hair brush/comb": "hairbrush comb",
    "stove/hob": "stove",
    "backyard": "backyard",
    "wheel barrow": "wheelbarrow",
    "boat": "boat",
}

country_dict = {
    "WestAsia": ["Turkey", "United Arab Emirates", "Saudi Arabia", "Jordan"],
    "SouthEastAsia": ["Indonesia", "Philippines", "Thailand", "Malaysia"],
    "Europe": ["Italy", "Spain", "United Kingdom", "Romania"],
    "EastAsia": ["Japan", "South Korea", "China", "Hong Kong"],
    "Americas": ["Colombia", "Argentina", "Mexico", "Brazil"],
    "Africa": ["Nigeria", "South Africa", "Egypt", "Angola"],
}

if which_dataset == "geode":
    # real geode data
    dataset_df = pd.read_pickle("geode_all_inceptionv3.pkl")
    dataset_df["features"] = [list(i) for i in dataset_df["features"]]
    dataset_df["id"] = dataset_df["file_path"]
    dataset_df["str_features"] = [
        " ".join([str(i) for i in k]) for k in dataset_df["features"]
    ]
    dataset_df["r"] = dataset_df["region"]
    deduped_dataset_df = dataset_df
    deduped_dataset_df = utils.get_df_deduped(dataset_df, keep_one=False)
    object_key = "object"
elif which_dataset == "ds":
    class_list = pd.read_csv("class_pivot_all_reasonable_classes.csv", index_col=0)
    class_list = class_list[class_list["Include?"] == 1]
    class_list = class_list.reset_index()
    class_list["object"] = [i.lower().replace("_", " ") for i in class_list["topics"]]

    dataset_df = pd.read_pickle("dollarstreet_all_inceptionv3.pkl")
    dataset_df["features"] = [list(i) for i in dataset_df["features"]]
    dataset_df["id"] = dataset_df["file_path"]
    dataset_df["str_features"] = [
        " ".join([str(i) for i in k]) for k in dataset_df["features"]
    ]
    dataset_df["r"] = (
        dataset_df["region.id"]
        .replace("af", "Africa")
        .replace("am", "Americas")
        .replace("as", "Asia")
        .replace("eu", "Europe")
    )
    dataset_df["object_reformatted"] = [
        i.lower().replace("_", " ") for i in dataset_df["topics"]
    ]
    dataset_df = dataset_df[dataset_df["object_reformatted"].isin(class_list["object"])]
    deduped_dataset_df = utils.get_df_deduped(dataset_df, keep_one=False)
    deduped_dataset_df = deduped_dataset_df[
        deduped_dataset_df["object_reformatted"] != "bed kids"
    ]
    deduped_dataset_df = deduped_dataset_df.drop(
        [36916]
    )  # for some reason this image doesn't have features
    object_key = "object_reformatted"

print(f"Data: {deduped_dataset_df.shape}")


def update_ds_df_features(model_df, geode_df_path):
    # update ds dataframe to include features from geode, to save on generations
    # geode df from model A
    model_df_geode = pd.read_pickle(geode_df_path)
    model_df_geode_dup = model_df_geode.copy(deep=True)
    updated_features = []
    for _, row in tqdm(model_df.iterrows()):
        if type(row["geode_df_path"]) == float:  # is na, then keep features
            updated_features.append(row["features"])
        else:
            hold = model_df_geode_dup[
                model_df_geode_dup["prompt"]
                == row["prompt"].replace(row["object"], obj_map_ds[row["object"]])
            ]
            drop_idx = hold.index[0]
            updated_features.append(hold["features"].loc[drop_idx])
            model_df_geode_dup.drop([drop_idx], inplace=True)
            # use `updated_features` from here on out
    model_df["updated_features"] = updated_features


#
# Load {object} in {region} and {object} model data
#
if which_dataset in ["geode", "ds"]:
    model_df = pd.read_pickle(
        f"{features_folder}/{which_dataset}_prompts_regions_fulldataset_{which_model}_{which_features}_deduped.pkl"
    )
    geode_model_df_path = f"{features_folder}/geode_prompts_regions_fulldataset_{which_model}_{which_features}_deduped.pkl"

    if which_dataset == "geode":
        model_df["r"] = (
            model_df["region"]
            .replace("the Americas", "Americas")
            .replace("East Asia", "EastAsia")
            .replace("Southeast Asia", "SouthEastAsia")
            .replace("West Asia", "WestAsia")
        )
        model_df["object"] = [i.replace(" ", "_") for i in model_df["object"]]
        features_key = "features"
    elif which_dataset == "ds" and model_df is not None:
        update_ds_df_features(model_df, geode_model_df_path)
        features_key = "updated_features"
        model_df = model_df[model_df["object"] != "bed kids"]
        model_df["r"] = model_df["region"].replace("the Americas", "Americas")
    if model_df is not None:
        model_df["id"] = model_df["img_id"]
        model_df[features_key] = [list(i) for i in model_df[features_key]]

        print(f"Fake: {model_df.shape}")

    if which_dataset == "ds":
        # fill in region info for {object} prompt in order to construct region-sized manifolds later
        full_model_df = pd.read_pickle(
            geode_model_df_path.replace("geode", "ds").replace("_deduped", "")
        )
        first_region_idx = full_model_df[~full_model_df["region"].isna()].index[0]
        assert (
            first_region_idx == full_model_df.shape[0] / 2
        ), "assumes original {obj} in {reg} df and {obj} dfs are same size"

        pretend_region_for_manifold = [None] * model_df.shape[0]
        for _, row in model_df[model_df["region"].isna()].iterrows():
            pretend_region_for_manifold[_] = full_model_df.loc[_ + first_region_idx][
                "region"
            ].replace("the Americas", "Americas")

        model_df["pretend_region_for_manifold"] = pretend_region_for_manifold

#
# Load {object} in {country} data
#
if which_dataset in ["geode", "ds"]:
    model_df_country = pd.read_pickle(
        f"{features_folder}/{which_dataset}_prompts_countries_fulldataset_{which_model}_{which_features}_deduped.pkl"
    )
    geode_model_df_path = f"{features_folder}/geode_prompts_countries_fulldataset_{which_model}_{which_features}_deduped.pkl"

    if which_dataset == "geode" and model_df_country is not None:
        model_df_country["r"] = (
            model_df_country["region"]
            .replace("the Americas", "Americas")
            .replace("East Asia", "EastAsia")
            .replace("Southeast Asia", "SouthEastAsia")
            .replace("West Asia", "WestAsia")
        )
        model_df_country["country"] = [
            prompt.split(" in ")[1] for prompt in model_df_country["prompt"]
        ]
        model_df_country["object"] = [
            i.replace(" ", "_") for i in model_df_country["object"]
        ]
        features_key = "features"
    elif which_dataset == "ds" and model_df_country is not None:
        update_ds_df_features(model_df_country, geode_model_df_path)
        features_key = "updated_features"
        model_df_country = model_df_country[model_df_country["object"] != "bed kids"]
        model_df_country["r"] = model_df_country["region"].replace(
            "the Americas", "Americas"
        )

    if model_df_country is not None:
        model_df_country[features_key] = [
            list(i) for i in model_df_country[features_key]
        ]
        model_df_country["id"] = model_df_country["img_id"]

        print(f"Fake C: {model_df_country.shape}")

# Prepare real and fake dataframes
object_file = "objects_180.txt"
real_hold = deduped_dataset_df
fake_hold_r = model_df[model_df.r.isin(list(set(real_hold.r)))]
if model_df_country is not None:
    fake_hold_c = model_df_country[model_df_country.r.isin(list(set(real_hold.r)))]
else:
    fake_hold_c = None

fake_hold = model_df[model_df.r.isna()]
fake_hold["r"] = ["-" for i in fake_hold["r"]]

if apply_filt:
    fake_hold = utils.sample_df(fake_hold, object_file, 42, 1080)
    fake_hold_r = utils.sample_df(fake_hold_r, object_file, 42, 180)
    if fake_hold_c is not None:
        fake_hold_c = utils.sample_df(fake_hold_c, object_file, 42, 180)

    if fake_hold_c is not None:
        print(
            f"Real: {real_hold.shape}, fake: {fake_hold.shape}, fake R:{fake_hold_r.shape}, fake C: {fake_hold_c.shape}"
        )
    else:
        print(
            f"Real: {real_hold.shape}, fake: {fake_hold.shape}, fake R:{fake_hold_r.shape}"
        )
else:
    if fake_hold_c is not None:
        print(
            f"Real: {real_hold.shape}, fake: {fake_hold.shape}, fake R:{fake_hold_r.shape}, fake C: {fake_hold_c.shape}"
        )
    else:
        print(
            f"Real: {real_hold.shape}, fake: {fake_hold.shape}, fake R:{fake_hold_r.shape}"
        )


real_features = np.array(real_hold["features"].values.tolist())
fake_features = np.array(fake_hold[features_key].values.tolist())
fake_features_r = np.array(fake_hold_r[features_key].values.tolist())
if fake_hold_c is not None:
    fake_features_c = np.array(fake_hold_c[features_key].values.tolist())


def get_clip_score_percentile(df, percentile=0.1):
    # A function to compute an average of 10-percentiles CLIPScore for each object in a region.

    df2 = pd.DataFrame(
        {col: vals["clip_score"] for col, vals in df.groupby(["object"])}
    )
    clip_score = {}
    clip_score["overall"] = df2.quantile([percentile]).values.mean()
    df2 = pd.DataFrame(
        {col: vals["clip_score"] for col, vals in df.groupby(["region", "object"])}
    )

    for r in df["region"].unique().tolist():
        clip_score[r] = df2[r].quantile([percentile]).values.mean()

    return pd.from_dict(clip_score)


#
# PRDC functions
# adapted from https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
# made faster - computing big manifolds a single time + per object metrics
def get_grouped_prdc_and_objects(
    real_features,
    fake_features,
    real_filter,
    fake_filter,
    nearest_k,
    filter_match=False,
    verbose=False,
    gather_gen_for_recall=False,
):
    """Perform pr measurements for certain subsets of data
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        real_filter: filter numpy.ndarray([N]) for which real_features to include
        fake_filter: filter numpy.ndarray([N]) for which fake_features to include
        nearest_k: int
        filter_match: bool, require that the filters are the same between real and fake features
        verbose: include prints
    """
    r_filters = []
    f_filters = []
    p = []
    r = []
    d = []
    c = []

    if len(list(set(fake_filter))) == 1 or gather_gen_for_recall:
        print(
            f"Computing fake manifold, fake filter: -; Num fake: {fake_features.shape[0]}"
        )
        fake_nearest_neighbour_distances = mm.compute_nearest_neighbour_distances(
            fake_features, nearest_k
        )
        print(fake_nearest_neighbour_distances.shape[0])

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

            fake_features_sub = fake_features[fake_filter == f_filter]

            if verbose:
                print(f"real filter: {r_filter}; fake filter: {f_filter}")
                print(
                    "Num real: {} Num fake: {}".format(
                        real_features_sub.shape[0], fake_features_sub.shape[0]
                    )
                )

            if len(list(set(fake_filter))) > 1 and not gather_gen_for_recall:
                # if fake_features_sub.shape[0] < nearest_k + 1:
                #     continue
                fake_nearest_neighbour_distances = (
                    mm.compute_nearest_neighbour_distances(fake_features_sub, nearest_k)
                )

            distance_real_fake = mm.compute_pairwise_distance(
                real_features_sub, fake_features_sub
            )

            if verbose:
                print("Computing precision...")
            precision = (
                (
                    distance_real_fake
                    < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                ).any(axis=0)
                # .mean()
            )

            if gather_gen_for_recall:
                print("Recomputing pairwise distances for recall")
                distance_real_fake = mm.compute_pairwise_distance(
                    real_features_sub, fake_features
                )

            if verbose:
                print("Computing recall...")
            recall = (
                (
                    distance_real_fake
                    < np.expand_dims(fake_nearest_neighbour_distances, axis=0)
                ).any(axis=1)
                # .mean()
            )

            if verbose:
                print("Computing density...")
            density = (1.0 / float(nearest_k)) * (
                distance_real_fake
                < np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).sum(axis=0)

            if verbose:
                print("Computing density...")
            coverage = distance_real_fake.min(axis=1) < real_nearest_neighbour_distances

            r_filters.append(r_filter)
            f_filters.append(f_filter)
            p.append(precision.mean())
            r.append(recall.mean())
            d.append(density.mean())
            c.append(coverage.mean())

    return pd.DataFrame(
        zip(r_filters, f_filters, p, r, d, c),
        columns=[
            "real_filter",
            "fake_filter",
            "precision",
            "recall",
            "density",
            "coverage",
        ],
    )


def get_grouped_prdc_perobject(
    real_features,
    fake_features,
    real_filter,
    fake_filter,
    real_object_filter,
    fake_object_filter,
    nearest_k,
    filter_match=False,
    verbose=True,
    img_id_real=None,
    img_id_fake=None,
):
    """Perform pr measurements for certain subsets of data
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        real_filter: region filter numpy.ndarray([N]) for which real_features to include
        fake_filter: region filter numpy.ndarray([N]) for which fake_features to include
        real_object_filter: object filter numpy.ndarray([N]) for which real_features to include
        fake_object_filter: object filter numpy.ndarray([N]) for which real_features to include
        nearest_k: int
        filter_match: bool, require that the filters are the same between real and fake features
        verbose: include prints
    """
    r_filters = []
    f_filters = []
    o_filters = []
    p = []
    r = []
    d = []
    c = []
    density_mat_or = []
    coverage_mat_or = []
    img_ids_density = []
    img_ids_coverage = []

    for r_filter in list(set(real_filter)):
        for o_filter in list(set(real_object_filter)):
            real_features_sub_sub = real_features[
                (real_object_filter == o_filter) * (real_filter == r_filter)
            ]
            ids = img_id_real[
                (real_object_filter == o_filter) * (real_filter == r_filter)
            ]
            img_ids_coverage.append(ids.tolist())
            # print(f"Computing real manifold, real filnter: {r_filter} {o_filter}; Num real: {real_features_sub.shape[0]}")

            real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
                real_features_sub_sub, nearest_k
            )

            for f_filter in list(set(fake_filter)):
                if filter_match and (r_filter != f_filter):
                    continue

                fake_features_sub_sub = fake_features[
                    (fake_object_filter == o_filter) * (fake_filter == f_filter)
                ]
                ids = img_id_fake[
                    (fake_object_filter == o_filter) * (fake_filter == f_filter)
                ]
                img_ids_density.append(ids.tolist())

                if verbose:
                    print(
                        f"real filter: {r_filter} {o_filter}; fake filter: {f_filter} {o_filter}"
                    )
                    print(
                        "Num real: {} Num fake: {}".format(
                            real_features_sub_sub.shape[0],
                            fake_features_sub_sub.shape[0],
                        )
                    )

                # if fake_features_sub_sub.shape[0] <= nearest_k + 1:
                #     continue
                fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
                    fake_features_sub_sub, nearest_k
                )

                distance_real_fake = compute_pairwise_distance(
                    real_features_sub_sub, fake_features_sub_sub
                )

                if verbose:
                    print("Computing precision...")
                precision = (
                    (
                        distance_real_fake
                        < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                    )
                    .any(axis=0)
                    .mean()
                )

                if verbose:
                    print("Computing recall...")
                recall = (
                    (
                        distance_real_fake
                        < np.expand_dims(fake_nearest_neighbour_distances, axis=0)
                    )
                    .any(axis=1)
                    .mean()
                )

                if verbose:
                    print("Computing density...")
                density = (1.0 / float(nearest_k)) * (
                    distance_real_fake
                    < np.expand_dims(real_nearest_neighbour_distances, axis=1)
                ).sum(axis=0)

                density_mat_or.append(density.tolist())

                density = density.mean()

                if verbose:
                    print("Computing coverage...")
                coverage = (
                    distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
                )

                coverage_mat_or.append(coverage.tolist())
                coverage = coverage.mean()

                o_filters.append(o_filter)
                r_filters.append(r_filter)
                f_filters.append(f_filter)
                p.append(precision)
                r.append(recall)
                d.append(density)
                c.append(coverage)

    if img_id_real is not None and img_id_fake is not None:
        return pd.DataFrame(
            zip(
                r_filters,
                f_filters,
                o_filters,
                p,
                r,
                d,
                c,
                density_mat_or,
                coverage_mat_or,
                img_ids_density,
                img_ids_coverage,
            ),
            columns=[
                "real_filter",
                "fake_filter",
                "object",
                "precision",
                "recall",
                "density",
                "coverage",
                "density_mat_or",
                "coverage_mat_or",
                "img_id_density",
                "img_id_coverage",
            ],
        )
    else:
        return pd.DataFrame(
            zip(r_filters, f_filters, o_filters, p, r, d, c),
            columns=[
                "real_filter",
                "fake_filter",
                "object",
                "precision",
                "recall",
                "density",
                "coverage",
            ],
        )


# #######
# #######
# ## Precision/recall for {object} generations vs. real data
# #######
# #######
print(f">>>>>>> Computing {which_metric} for object prompt.")
if which_metric == "pergroup_pr":
    if apply_filt:
        fake_hold_sub = utils.sample_df(fake_hold, object_file, 42, 180)
        fake_features_sub = np.array(fake_hold_sub[features_key].values.tolist())
        filter_match = False
    else:
        fake_hold_sub = fake_hold_r  # to get regions
        fake_features_sub = fake_features
        filter_match = True

    if fake_hold_sub.shape[0] == len(fake_features_sub):
        print("using region standin")
    pr_df_all_g = get_grouped_prdc_and_objects(
        real_features,
        fake_features_sub,
        real_hold["r"],
        fake_hold_sub["r"]
        if fake_hold_sub.shape[0] == len(fake_features_sub)
        else fake_hold[
            "pretend_region_for_manifold"
        ],  # to emulate sampling following region-wise object distribution using region df if the sizes match or populated placeholder region if not.
        nearest_k=k_PR,
        filter_match=filter_match,
        verbose=True,
        gather_gen_for_recall=False,
    )
    pr_df_all_g.to_csv(
        f"{save_metrics_path}object_pergroup_pr_{which_model}_{which_features}_{k_PR}{ext}.csv"
    )

if which_metric == "perobj_pergroup_pr":
    if apply_filt:
        fake_hold_sub = utils.sample_df(fake_hold, object_file, 42, 180)
        fake_features_sub = np.array(fake_hold_sub[features_key].values.tolist())
        filter_match = False
    else:
        fake_hold_sub = fake_hold_r  # to get regions
        fake_features_sub = fake_features
        filter_match = True

    pr_df_all_g_obj = get_grouped_prdc_perobject(
        real_features,
        fake_features_sub,
        real_hold["r"],
        fake_hold_sub[
            "r"
        ],  # to emulate sampling following region-wise object distribution
        real_hold[object_key],
        fake_hold_sub["object"],
        nearest_k=k_PR,
        filter_match=filter_match,
        img_id_real=real_hold["file_path"],
        img_id_fake=fake_hold_sub["img_id"],
    )
    pr_df_all_g_obj.to_csv(
        f"{save_metrics_path}object_pergroup_pr_perobject_{which_model}_{which_features}_{k_PR}{ext}.csv"
    )


#######
######
## Precision/recall for {object} in {region} generations vs. real data
#######
#######
# all real vs all {object} in {region}
print(f">>>>>>> Computing {which_metric} for object in region prompt.")

if which_metric == "pergroup_pr":
    pr_df_region_g = get_grouped_prdc_and_objects(
        real_features,
        fake_features_r,
        real_hold["r"],
        fake_hold_r["r"],
        nearest_k=k_PR,
        filter_match=True,
        verbose=True,
        gather_gen_for_recall=False,
    )
    pr_df_region_g.to_csv(
        f"{save_metrics_path}object_region_pergroup_pr_{which_model}_{which_features}_{k_PR}{ext}.csv"
    )

if which_metric == "perobj_pergroup_pr":
    pr_df_region_g_obj = get_grouped_prdc_perobject(
        real_features,
        fake_features_r,
        real_hold["r"],
        fake_hold_r["r"],
        real_hold["object"],
        fake_hold_r["object"],
        nearest_k=k_PR,
        filter_match=True,
        img_id_real=real_hold["file_path"],
        img_id_fake=fake_hold_r["img_id"],
    )
    pr_df_region_g_obj.to_csv(
        f"{save_metrics_path}object_region_pergroup_pr_perobject_{which_model}_{which_features}_{k_PR}{ext}.csv"
    )


# ######
# ######
# ## Precision/recall for {object} in {country} generations vs. real data
# ######
# ######
print(f">>>>>>> Computing {which_metric} for object in country prompt.")

if fake_hold_c is not None:
    fake_objects_c = np.array(fake_hold_c["object"].values.tolist())

    if which_metric == "pergroup_pr":
        pr_df_country_g = get_grouped_prdc_and_objects(
            real_features,
            fake_features_c,
            real_hold["r"],
            fake_hold_c["r"],
            nearest_k=k_PR,
            filter_match=True,
            verbose=True,
            gather_gen_for_recall=False,
        )
        pr_df_country_g.to_csv(
            f"{save_metrics_path}object_country_pergroup_pr_{which_model}_{which_features}_{k_PR}{ext}.csv"
        )

    if which_metric == "perobj_pergroup_pr":
        pr_df_country_g_obj = get_grouped_prdc_perobject(
            real_features,
            fake_features_c,
            real_hold["r"],
            fake_hold_c["r"],
            real_hold[object_key],
            fake_hold_c["object"],
            nearest_k=k_PR,
            filter_match=True,
            img_id_real=real_hold["file_path"].values,
            img_id_fake=fake_hold_c["img_id"].values,
        )
        pr_df_country_g_obj.to_csv(
            f"{save_metrics_path}object_country_pergroup_pr_perobject_{which_model}_{which_features}_{k_PR}{ext}.csv"
        )


if which_metric == "clip_score":
    assert which_features == "clip_score"

    clip_score = get_clip_score_percentile(fake_hold)
    clip_score.to_csv(f"{save_metrics_path}clip_scores_object_{which_model}_{ext}.csv")

    clip_score = get_clip_score_percentile(fake_hold_r)
    clip_score.to_csv(f"{save_metrics_path}clip_scores_region_{which_model}_{ext}.csv")

    if fake_hold_c is not None:
        clip_score = get_clip_score_percentile(fake_hold_c)

        clip_score.to_csv(
            f"{save_metrics_path}clip_scores_country_{which_model}_{ext}.csv"
        )
