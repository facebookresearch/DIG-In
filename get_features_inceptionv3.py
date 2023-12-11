# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# MIT License

import pandas as pd
from torchvision.io import read_image
import torch.nn as nn
import torch
from torchvision.models.inception import inception_v3
import torch.nn.functional as F
from torch.nn import Parameter as P
from PIL import Image

from tqdm import tqdm
import numpy as np

import utils

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("-which_dataset", "--which_dataset", type=str)
parser.add_argument("-prompt_df_path", "--prompt_df_path", type=str)
parser.add_argument("-img_path", "--img_path", type=str)
parser.add_argument("-save_path", "--save_path", type=str)
parser.add_argument("-geode_path", "--geode_path", type=str)
args = parser.parse_args()


# from https://github.com/facebookresearch/ic_gan/blob/8eff2f7390e385e801993211a941f3592c50984d/data_utils/inception_utils.py
# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = P(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        # Normalize x
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        # 1 x 1 x 2048
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        # 1000 (num_classes)
        return pool, logits


# Load and wrap the Inception model
def load_inception_net(parallel=False, device="cuda"):
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = WrapInception(inception_model.eval()).to(device)
    if parallel:
        print("Parallelizing Inception module...")
        inception_model = nn.DataParallel(inception_model)
    return inception_model


def prompt_features(prompt_df_path, img_path, save_path, geode_path=None):
    net = load_inception_net(parallel=False, device="cuda")
    # Center crop, resize to 299, convert to [0, 1] tensor, and normalize
    preprocess = utils.get_preprocess()

    df = pd.read_csv(prompt_df_path)
    features = []
    for _, row in tqdm(df.iterrows()):
        path = f'{img_path}{row["prompt"].replace("/", "_")}__{row["img_id"]}.png'
        if "geode_df_filename" in row.keys() and type(row["geode_df_filename"]) == str:
            path = f'{geode_path}{row["geode_df_filename"]}'

        if row["object"] == "bed kids":
            features.append(None)
        else:
            with torch.no_grad():
                features.append(
                    net(preprocess(Image.open(path)).to("cuda").float())[0]
                    .cpu()
                    .detach()
                    .numpy()[0]
                )

    df["features"] = features
    df.to_pickle(save_path)


if args.which_dataset == "generated":
    geode_path = None if args.geode_path == "" else args.geode_path
    prompt_features(args.prompt_df_path, args.img_path, args.save_path, geode_path)

    model_df = pd.read_pickle(f"{args.save_path}")
    print(model_df.shape)

    model_df["id"] = model_df["img_id"]

    model_df["str_features"] = model_df["features"].apply(str)
    model_df_deduped = utils.get_df_deduped(model_df, keep_one=False)

    model_df_deduped.to_pickle(f"{args.save_path[:-4]}_deduped.pkl")
