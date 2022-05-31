from abc import abstractmethod
from typing import List
try:
    from typing import final
except ImportError:  # Python version < 3.8
    final = lambda f: f  # Identity decorator

from pytorch_lightning import LightningModule
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from graphnet.models.graph_builders import GraphBuilder


class Detector(LightningModule):
    """Base class for all detector-specific read-ins in graphnet."""

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """List of features used/assumed by inheriting `Detector` objects."""

    def __init__(self, graph_builder: GraphBuilder, scalers: List[dict] = None):
        # Base class constructor
        super().__init__()

        # Member variables
        self._graph_builder = graph_builder
        self._scalers = scalers
        if self._scalers:
            print("Will use scalers rather than standard preprocessing",
                 f"in {self.__class__.__name__}.")

    @final
    def forward(self, data: Data) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""

        # Check(s)
        assert data.x.size()[1] == self.nb_inputs, \
            ("Got graph data with incompatible size, ",
            f"{data.x.size()} vs. {self.nb_inputs} expected")

        # Graph-bulding
        # @NOTE: `.clone` is necessary to avoid modifying original tensor in-place.
        data = self._graph_builder(data).clone()
        # print(data.edge_index)
        # print(data.edge_attr)
        # print(data.x)
        # exit()

        if self._scalers:
            # # Scaling individual features
            # x_numpy = data.x.detach().cpu().numpy()
            # for key, scaler in self._scalers.items():
            #     ix = self.features.index(key)
            #     data.x[:,ix] = torch.tensor(scaler.transform(x_numpy[:,ix])).type_as(data.x)

            # Scaling groups of features | @TEMP, probably
            x_numpy = data.x.detach().cpu().numpy()

            data.x[:,:3] = torch.tensor(
                self._scalers['xyz'].transform(x_numpy[:,:3])
            ).type_as(data.x)

            data.x[:,3:] = torch.tensor(
                self._scalers['features'].transform(x_numpy[:,3:])
            ).type_as(data.x)

        else:
            # Implementation-specific forward pass (e.g. preprocessing)
            data = self._forward(data)

        return data

    @abstractmethod
    def _forward(self, data: Data) -> Data:
        """Same syntax as `.forward` for implentation in inheriting classes."""

    @property
    def nb_inputs(self) -> int:
        return len(self.features)

    @property
    def nb_outputs(self) -> int:
        """This the default, but may be overridden by specific inheriting classes."""
        return self.nb_inputs

    def _validate_features(self, data: Data):
        if isinstance(data, Batch):
            data_features = data[0].features
        else:
            data_features = data.features
        assert data_features == self.features, \
            "Features on Data and Detector differ: {data_features} vs. {self.features}"
