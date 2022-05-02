import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from torch.optim.adam import Adam

from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder


import common


def train_test(args: common.Args, get_dataloaders, get_parts):
    print()
    print(f'===== train({args.run_name=}, {args.target=}) =====')
    print(f'{args.features=}')
    print(f'{args.truth=}')

    # Getting data
    training_dataloader, validation_dataloader, test_dataloader = \
        get_dataloaders(args)

    # Building model
    parts = get_parts(args, training_dataloader)
    model = parts.model

    # Setup Training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        # logger=wandb_logger,
    )

    # Training
    trainer.fit(model, training_dataloader, validation_dataloader)

    # Predicting
    results = common.get_predictions(
        target=args.target,
        trainer=trainer,
        model=model,
        test_dataloader=test_dataloader,
    )

    # Saving to file
    os.makedirs(args.archive.root_str, exist_ok=True)
    model.save_state_dict(args.archive.state_dict_str)
    model.save(args.archive.model_str)
    results.to_csv(args.archive.results_str)
