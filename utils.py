# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import torchvision.transforms as transforms
from torchvision.io import read_image
import random
import pickle
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
import clip


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    From: https://github.com/facebookresearch/ic_gan/blob/8eff2f7390e385e801993211a941f3592c50984d/data_utils/utils.py
    Parameters
    ----------
        size: sequence or int
            Desired output size of the crop. If size is an int instead of sequence like (h, w),
            a square crop (size, size) is made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def get_df_deduped(df, keep_one=False):
    """Address imagesa with identical features.

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

    return df[df["id"].isin(ids)]


def sample_df(df, object_file, seed, n_img, replacement=False, objects=None):
    """Sample DF to ensure same number of images per object - concept combination.
    df to sample, w columns ['object', 'r']
    object_file containing one object per line
    seed for sampling
    n_img per object - region combination
    """
    if objects == None:
        with open(object_file, "rb") as fp:
            objects = pickle.load(fp)
    idxs = []
    regions = set(df["r"])
    for o in objects:
        for r in regions:
            hold = df[(df["object"] == o) & (df["r"] == r)]
            random.seed(seed)
            if replacement:
                idxs.extend(random.choices(list(hold.index), k=n_img))
            else:
                idxs.extend(random.sample(list(hold.index), n_img))

    return df.loc[idxs]


def get_preprocess():
    """Preprocess images to support feature extraction from an Inception V3.

    Adapted from https://github.com/facebookresearch/ic_gan/blob/8eff2f7390e385e801993211a941f3592c50984d/data_utils/utils.py#L403-L413.
    The normalization assumes values are [0, 1] prior to normalization, which occurs in ToTensor.
    """
    # Data transforms
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    transform_list = transforms.Compose(
        [
            CenterCropLongEdge(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    return transform_list


def init_clip_score():
    clip_score = {}
    clip_score["o"] = []
    clip_score["r"] = []
    clip_score["c"] = []
    clip_score["img_path"] = []

    # Load the pre-trained CLIP model and the image
    model, preprocess = clip.load("ViT-B/32")
    # model, preprocess = clip.load('ViT-L/14')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return clip_score, model, preprocess


def compute_clip_score(image, text, model, preprocess):
    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    # print(text_input)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    # print(image_features.shape, text_features.T.shape)
    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score
