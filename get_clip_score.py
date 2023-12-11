# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
import clip
from PIL import Image
import torch
from tqdm import tqdm
from utils import compute_clip_score


def prompt_features(prompt_df_path, img_path, save_path):
    missing_prompts = []
    found_prompts = []
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    print(prompt_df_path)
    df = pd.read_csv(prompt_df_path)
    scores = []
    for _, row in tqdm(df.iterrows()):
        with torch.no_grad():
            try:
                scores.append(
                    compute_clip_score(
                        Image.open(f'{img_path}{row["prompt"]}__{row["img_id"]}.png'),
                        row["object"],
                        model=model,
                        preprocess=preprocess,
                    )
                )
                found_prompts.append(row["prompt"])
            except:
                missing_prompts.append(row["prompt"])

    df["clip_score"] = scores
    df.to_pickle(save_path)


def get_df_deduped(df, keep_one=False):
    """Address images with identical features.

    Parameters:
        df (Pandas DataFrame): Contains columns with
            `id` (str) unique,
            `str_features` (str) (to be de-duplicated),
            `r` (str).
        keep_one (bool): Should we keep one instance or remove all?

    Returns:
        deduped_df (Pandas DataFrame): Unique images.
    """
    assert len(set(df.id)) == len(df.id)

    before = len(df.id)

    grouped_df = df[["id", "str_features"]].groupby(["str_features"])["id"].apply(list)
    grouped_df = grouped_df.reset_index()

    unique_feature_ids = list(grouped_df["id"])

    ids = []
    for i in unique_feature_ids:
        if len(i) > 1:
            if keep_one:
                # only include image if all duplicates have the same region
                if len(set(df[df["id"].isin(i)]["r"])) == 1:
                    ids.append(i[0])
        else:
            ids.append(i[0])

    after = len(ids)
    print(f"Removed {before-after} features.")

    return df[df["id"].isin(ids)]


def main(prompt_df_path, img_path, save_path):
    prompt_features(prompt_df_path, img_path, save_path)

    model_df = pd.read_pickle(f"{save_path}")
    print(model_df.shape)

    model_df["id"] = model_df["img_id"]

    model_df["str_features"] = model_df["features"].apply(str)
    model_df_deduped = get_df_deduped(model_df, keep_one=False)

    model_df_deduped.to_pickle(f"{save_path[:-4]}_deduped.pkl")


if __name__ == "__main__":
    prompt_df_path = "geode_prompts_regions_fulldataset.csv"
    img_path = ""  # TODO -- directory of images following the prompt_df_path structure
    save_path = ""  # TODO -- where to save features

    main(prompt_df_path, img_path, save_path)
