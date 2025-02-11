from typing import Optional

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_scatter import scatter


def sum_pool_and_distribute(tensor: Tensor, cluster_index: LongTensor, batch: Optional[LongTensor] = None) -> Tensor:
    """Sum-pool values across the cluster, and distribute the individual nodes."""
    if batch is None:
        batch = torch.zeros(tensor.size(dim=0)).long()
    tensor_pooled, _ = sum_pool_x(cluster_index, tensor, batch)
    inv, _ = consecutive_cluster(cluster_index)
    tensor_unpooled = tensor_pooled[inv]
    return tensor_unpooled


def group_identical(tensor: Tensor, batch: Optional[LongTensor] = None) -> Tensor:
    """Group rows in `tensor` that are identical

    Args:
        tensor (Tensor): Tensor of shape [N, F]
        batch (Optional[LongTensor], optional): Batch indices, to only group
            identical rows within batches. Defaults to None.

    Returns:
        Tensor: List of group indices, from 0 to num. groups - 1, assigning all
            identical rows to the same group.
    """
    if batch is not None:
        tensor = tensor.cat((tensor, batch.unsqueeze(dim=1)), dim=1)
    return torch.unique(tensor, return_inverse=True, dim=0)[1]

def group_pulses_to_dom(data: Data) -> Data:
    """Groups pulses on the same DOM, using DOM and string number."""
    tensor = torch.stack((data.dom_number, data.string)).T.int()
    batch = getattr(tensor, 'batch', None)
    data.dom_index = group_identical(tensor, batch)
    return data

def group_pulses_to_pmt(data: Data) -> Data:
    """Groups pulses on the same PMT, using PMT, DOM, and string number."""
    tensor = torch.stack((data.pmt_number, data.dom_number, data.string)).T.int()
    batch = getattr(tensor, 'batch', None)
    data.pmt_index = group_identical(tensor, batch)
    return data


# Below mirroring `torch_geometric.nn.pool.{avg,max}_pool.py` exactly
def _sum_pool_x(cluster, x, size: Optional[int] = None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='sum')


def sum_pool_x(cluster, x, batch, size: Optional[int] = None):
    r"""Sum-Pools node features according to the clustering defined in
    :attr:`cluster`.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`LongTensor`) if :attr:`size` is
        :obj:`None`, else :class:`Tensor`
    """
    if size is not None:
        batch_size = int(batch.max().item()) + 1
        return _sum_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _sum_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def sum_pool(cluster, data, transform=None):
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *sum* of features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)

    x = None if data.x is None else _sum_pool_x(cluster, data.x)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data
