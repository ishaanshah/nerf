import json
import numpy as np
import torchvision.transforms.functional as F
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation
from typing import Tuple
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class NeRFDataSet(Dataset):
    def __init__(self, mode: str, data_dir: Path):
        self.mode = mode
        self.data_dir = data_dir
        with open(data_dir / f"transforms_{mode}.json") as f:
            self.frames = json.load(f)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO: Sample single pixel from image not entire image
        mat = np.asarray(self.frames[idx]["transform_matrix"])

        # Get position
        pos = mat[:3,3]

        # Get rotation
        ang = Rotation.from_matrix(mat[:3,:3]).as_euler('xyz')

        # Prepare input tensor
        inp = torch.Tensor([*pos, ang[0], ang[2]])

        # Convert image to tensor
        img = F.to_tensor(Image.open(self.frames[idx]["file_path"]))

        return inp, img

    def __len__(self) -> int:
        return len(self.frames["frames"])


class NeRFDataModule(LightningDataModule):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def prepare_data(self):
        # TODO: Write download code here
        raise NotImplemented()

    def train_dataloader(self):
        dataset = NeRFDataSet(mode="train", data_dir=self.data_dir)
        return DataLoader(dataset, batch_size=4096, shuffle=True)

    def val_dataloader(self):
        dataset = NeRFDataSet(mode="val", data_dir=self.data_dir)
        return DataLoader(dataset, batch_size=4096, shuffle=True)

    def test_dataloader(self):
        dataset = NeRFDataSet(mode="test", data_dir=self.data_dir)
        return DataLoader(dataset, batch_size=4096, shuffle=True)
