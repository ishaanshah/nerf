import torchmetrics as metrics
import wandb
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, Optional, List
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from pathlib import Path
from . import utils
from .model import NeRFModel
from .dataset import NeRFBlenderDataSet


class NeRFModule(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("NeRF")
        parser.add_argument(
            "--lx",
            type=int,
            default=10,
            help="number of components to use for encoding position",
        )
        parser.add_argument(
            "--ld",
            type=int,
            default=4,
            help="number of components to use for encoding direction",
        )
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument(
            "--sample_coarse", type=int, default=64, help="number of coarse samples"
        )
        parser.add_argument(
            "--sample_fine",
            type=int,
            default=192,
            help="number of fine samples excluding coarse samples",
        )
        return parent_parser

    def __init__(
        self, args: Namespace, wandb_logger: Optional[WandbLogger] = None
    ) -> None:
        super(NeRFModule, self).__init__()
        self.args = args
        self.wandb_logger = wandb_logger
        self.save_hyperparameters()

        # Models
        self.model_coarse = NeRFModel(args.lx * 6, args.ld * 6)
        self.model_fine = NeRFModel(args.lx * 6, args.ld * 6)

        # Loss
        self.criterion = nn.MSELoss(reduction="mean")

        # Metrics
        self.psnr = metrics.PeakSignalNoiseRatio()
        self.ssim = metrics.StructuralSimilarityIndexMeasure()

        # Create datasets
        data_dir = Path(args.data_dir)
        self.train_dataset = NeRFBlenderDataSet(
            mode="train", data_dir=data_dir, scale=self.args.scale
        )
        self.val_dataset = NeRFBlenderDataSet(
            mode="val", data_dir=data_dir, scale=self.args.scale
        )

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Inputs -
            batch:
                o [B * 3]: Origin of rays
                d [B * 3]: Direction of rays
                c [B * 3]: Target color for the ray
                near/far [B]: Bounds of scene

        Outputs -
            loss: MSELoss between predicted and target color
        """
        o, d, _, near, far = batch
        B = o.shape[0]

        # Get coarse color
        t_coarse = utils.sample_coarse(B, self.args.sample_coarse, near, far)

        # TODO: Get fine color
        return utils.render(
            o,
            d,
            t_coarse,
            self.model_coarse,
            self.model_fine,
            self.args.lx,
            self.args.ld,
            self.train_dataset.white_bck,
        )

    def training_step(self, batch, _) -> Tensor:
        _, _, c, _, _ = batch

        cp, w = self(batch)
        c_loss = self.criterion(cp, c)

        # Logging
        # TODO: Fix SSIM
        psnr = self.psnr(cp, c)
        # ssim = self.ssim(cp, c)
        self.log("train/c_loss", c_loss)
        self.log("train/loss", c_loss)
        self.log("train/psnr", psnr, prog_bar=True)
        # self.log("train/ssim", ssim)

        return c_loss

    def validation_step(self, batch, _) -> dict:
        nbatch = [i.squeeze(0) for i in batch]
        _, _, c, _, _ = nbatch

        cp, w = self(nbatch)
        c_loss = self.criterion(cp, c)

        # Logging
        psnr = self.psnr(cp, c)
        # ssim = self.ssim(cp, c)
        self.log("val/c_loss", c_loss)
        self.log("val/loss", c_loss)
        self.log("val/psnr", psnr)
        # self.log("val/ssim", ssim)
        return {"pred": cp, "gt": c}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Log predicted and ground truth images"""
        if self.wandb_logger:
            columns = ["ground_truth", "predicted"]
            data = []
            # TODO: Fix this to be dataset agnostic
            w = int(np.floor(self.val_dataset.w))
            h = int(np.floor(self.val_dataset.h))
            for output in outputs:
                gt = output["gt"].reshape(h, w, 3).cpu().numpy()*255
                pred = output["pred"].reshape(h, w, 3).cpu().numpy()*255
                data.append([wandb.Image(gt), wandb.Image(pred)])

            self.wandb_logger.log_table(key="results", columns=columns, data=data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # TODO: Uncomment this later
    """
    def test_dataloader(self) -> DataLoader:
        dataset = NeRFBlenderDataSet(mode="test", data_dir=self.data_dir)
        return DataLoader(dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.model_coarse.parameters(), lr=self.args.lr, eps=1e-7
        )
        # TODO: Configure scheduler
        # decay_rate = 0.1
        # lrate_decay = self.args.lr_decay
        # func = lambda step: 1 / (1 + decay_rate*lrate_decay)
        # scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, gamma=0.99)
        return optimizer
