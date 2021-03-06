import json
import numpy as np
import torchvision.transforms.functional as F
import torch
import os
from pathlib import Path
from .utils import get_rays
from torch import Tensor
from typing import Tuple, List, Optional
from PIL import Image
from torch.utils.data import Dataset


class NeRFBlenderDataSet(Dataset):
    def __init__(
        self,
        mode: str,
        data_dir: Path,
        scale: float,
        valid_count: int = -1,
        test_count: int = -1,
        img_list: List[str] = [],
    ):
        """
        Inputs:
            mode: Partition of data to use (train, test, val)
            data_dir: Base directory to get files from
            scale: Image scaling
            valid/test_count: Number of frames to use for validation/test (all if -1)
            img_list: The frames to consider while training
        """
        self.mode = mode
        self.data_dir = data_dir
        self.valid_count = valid_count
        self.test_count = test_count
        self.white_bck = False
        self.img_list = img_list

        with open(data_dir / f"transforms_{mode}.json") as f:
            self.frames = json.load(f)

        # Dimensions of image
        self.w = int(np.floor(800 * scale))
        self.h = int(np.floor(800 * scale))

        # NOTE: Uncomment for center crop
        # self.wc = int(self.w*0.8)
        # self.hc = int(self.h*0.8)

        # Focal length
        self.f = (self.w / 2) * (1 / np.tan(self.frames["camera_angle_x"] / 2))

        # Near and far bounds
        self.near = 2.0
        self.far = 6.0

        # Directions are same in camera coordinates for all
        # NOTE: Uncomment for center crop
        # x, y = torch.meshgrid(torch.arange(self.wc), torch.arange(self.hc), indexing="xy")
        # self.dirs = torch.dstack(
        #     [(x - self.wc / 2) / self.f, -(y - self.hc / 2) / self.f, -torch.ones_like(y)]
        # )
        x, y = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.dirs = torch.dstack(
            [(x - self.w / 2) / self.f, -(y - self.h / 2) / self.f, -torch.ones_like(y)]
        )

        # Only sample random rays in train mode
        if self.mode == "train":
            origins = []
            directions = []
            rgbs = []
            for frame in self.frames["frames"]:
                if self.img_list and (
                    not int(os.path.basename(frame["file_path"]).split("_")[1])
                    in self.img_list
                ):
                    continue

                o, d, img = self.gen_from_frame(frame)
                origins.append(o)
                directions.append(d)
                rgbs.append(img)

            self.origins = torch.cat(origins, 0)
            self.directions = torch.cat(directions, 0)
            self.rgbs = torch.cat(rgbs, 0)

    def gen_from_frame(self, frame: dict) -> Tuple[Tensor, Tensor, Tensor]:
        # Generate rays
        mat = torch.Tensor(frame["transform_matrix"])
        o, d = get_rays(mat, self.dirs)
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)

        # Get RGB values
        img = Image.open(
            os.path.join(self.data_dir, f"{frame['file_path']}.png")
        ).convert("RGBA")
        bkg = Image.new("RGBA", img.size, (0, 0, 0))
        img = Image.alpha_composite(bkg, img).convert("RGB")
        img = img.resize((self.w, self.h))
        img = F.to_tensor(img).double()

        # NOTE: Uncomment for center crop
        # img = F.center_crop(img, [self.wc, self.hc])

        # NOTE: Uncomment for viewing input image (for debuging)
        # F.to_pil_image(img).save('debug.png')

        img = img.reshape(3, -1).t()
        return o, d, img

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, float, float]:
        if self.mode == "train":
            return (
                self.origins[idx],
                self.directions[idx],
                self.rgbs[idx],
                self.near,
                self.far,
            )
        else:
            frame = self.frames["frames"][idx]
            o, d, img = self.gen_from_frame(frame)
            return o, d, img, self.near, self.far

    def __len__(self) -> int:
        if self.mode == "train":
            return self.rgbs.shape[0]
        elif self.mode == "val":
            if self.valid_count < 0:
                return len(self.frames["frames"])
            return self.valid_count
        else:
            if self.test_count < 0:
                return len(self.frames["frames"])
            return self.test_count
