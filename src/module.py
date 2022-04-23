import torchmetrics as metrics
import wandb
import numpy as np
import torch
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
        parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
        parser.add_argument(
            "--sample_coarse", type=int, default=64, help="number of coarse samples"
        )
        parser.add_argument(
            "--sample_fine",
            type=int,
            default=192,
            help="number of fine samples excluding coarse samples",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=32 * 1024,
            help="number rays to process at a time",
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

        chunk = self.args.chunk_size
        colors_c, colors_f = [], []
        for i in range(0, B, chunk):
            # Coarse sampling
            t_coarse = utils.sample_coarse(
                len(o[i : i + chunk]), self.args.sample_coarse, near, far
            )

            c_c, w_c = utils.render(
                    o[i : i + chunk],
                    d[i : i + chunk],
                    t_coarse,
                    self.model_coarse,
                    self.args.lx,
                    self.args.ld,
                    len(o[i : i + chunk]),
                    self.train_dataset.white_bck,
                )

            colors_c += [c_c]

            # Fine sampling
            t_bins = (t_coarse[:,:-1] + t_coarse[:,1:]) / 2
            t_fine = utils.sample_fine(self.args.sample_fine, t_bins, w_c[:,1:-1]).detach()
            t_fine = torch.cat((t_fine, t_coarse), dim=1)
            t_fine = torch.sort(t_fine, dim=1)[0]
            c_f, _ = utils.render(
                    o[i : i + chunk],
                    d[i : i + chunk],
                    t_fine,
                    self.model_fine,
                    self.args.lx,
                    self.args.ld,
                    len(o[i : i + chunk]),
                    self.train_dataset.white_bck,
                )
            colors_f += [c_f]


        colors_coarse = torch.cat(colors_c, dim=0) # B * 3
        colors_fine = torch.cat(colors_f, dim=0) # B * 3

        return colors_coarse, colors_fine

    def training_step(self, batch, _) -> Tensor:
        _, _, c, _, _ = batch

        cc_p, cf_p = self(batch)
        c_loss = self.criterion(cc_p, c)
        f_loss = self.criterion(cf_p, c)
        loss = c_loss+f_loss

        # Logging
        # TODO: Fix SSIM
        with torch.no_grad():
            psnr = self.psnr(cf_p, c)
        # ssim = self.ssim(cp, c)
        self.log("train/c_loss", c_loss, prog_bar=True)
        self.log("train/f_loss", f_loss, prog_bar=True)
        self.log("train/loss", loss)
        self.log("train/psnr", psnr, prog_bar=True)
        # self.log("train/ssim", ssim)

        return loss

    def validation_step(self, batch, _) -> dict:
        nbatch = [i.squeeze(0) for i in batch]
        _, _, c, _, _ = nbatch

        cc_p, cf_p = self(nbatch)
        c_loss = self.criterion(cc_p, c)
        f_loss = self.criterion(cf_p, c)
        loss = c_loss+f_loss

        # Logging
        # TODO: Fix SSIM
        with torch.no_grad():
            psnr = self.psnr(cf_p, c)
        # ssim = self.ssim(cp, c)
        self.log("val/c_loss", c_loss)
        self.log("val/f_loss", f_loss)
        self.log("val/loss", loss)
        self.log("val/psnr", psnr)
        # self.log("val/ssim", ssim)

        return {"pred_fine": cf_p, "pred_coarse": cc_p, "gt": c}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Log predicted and ground truth images"""
        if self.wandb_logger:
            columns = ["id", "ground_truth", "predicted"]
            data = []
            w = int(np.floor(self.val_dataset.w))
            h = int(np.floor(self.val_dataset.h))
            for i in range(0, min(5, len(outputs))):
                output = outputs[i]
                gt = output["gt"].reshape(h, w, 3).cpu().numpy() * 255
                pred = output["pred_fine"].reshape(h, w, 3).cpu().numpy() * 255
                data.append([i, wandb.Image(gt), wandb.Image(pred)])

            self.wandb_logger.log_table(key="rgb", columns=columns, data=data)

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
            params=list(self.model_coarse.parameters()) + list(self.model_fine.parameters()), lr=self.args.lr, eps=1e-7
        )
        # TODO: Configure scheduler
        # decay_rate = 0.1
        # lrate_decay = self.args.lr_decay
        # func = lambda step: 1 / (1 + decay_rate*lrate_decay)
        # scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, gamma=0.99)
        return optimizer
