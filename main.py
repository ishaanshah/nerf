from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.module import NeRFModule


def main(args):
    wandb_logger = None

    # Create the trainer
    if args.logger == False:
        logger = False
    else:
        # Create logger object
        wandb_logger = WandbLogger(project="nerf")
        logger = wandb_logger
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
    )

    # Create the model
    model = NeRFModule(args, wandb_logger)

    # Train the model
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Program specific arguments
    parser.add_argument("data_dir", type=str, help="path to dataset")
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="batch size to train with"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="scaling factor for images"
    )
    parser.add_argument(
        "--valid_count",
        type=int,
        default=-1,
        help="how many images to use for validation (-1 for all)",
    )
    parser.add_argument(
        "--img_list",
        type=str,
        default="",
        help="indices of subset of images to consider while training",
    )

    # Model specific arguments
    parser = NeRFModule.add_model_specific_args(parser)

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # Parse all arguments
    args = parser.parse_args()

    main(args)
