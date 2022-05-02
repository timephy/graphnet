import common

from torch.optim.adam import Adam
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR


get_dataloaders = common.get_dataloaders


def get_parts(args: common.Args, training_dataloader, *, nb_nearest_neighbours=8) -> common.Values:
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=nb_nearest_neighbours)
    )
    gnn = DynEdge(nb_inputs=detector.nb_outputs)
    task = common.get_task(target=args.target, gnn=gnn)

    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * args.max_epochs],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },
    )

    return common.Values(
        detector=detector,
        gnn=gnn,
        task=task,
        model=model,
    )
