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
        log_every_n_steps=10,
        # profiler='simple',
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

    # Model specific arguments
    parser = NeRFModule.add_model_specific_args(parser)

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # Parse all arguments
    args = parser.parse_args()

    main(args)
