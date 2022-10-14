from typing import Optional, Callable
from torch import nn as nn
import torch_geometric as pyg
from torch_geometric import nn as pyg_nn
import random
from itertools import product
from collections import defaultdict
import torch.nn.functional as F
import torch
import collate as colt
from torch_geometric.data import InMemoryDataset, Data, HeteroData
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
from torch.utils import data
import math
from torch_geometric.loader import DataLoader
import Edgeconv_f as egf
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from torch_geometric.nn.inits import reset

try:
    from torch_cluster import knn
except ImportError:
    knn = None

#######################################################################################################################
'''函数'''
def get_num_nodes(avg_num_nodes: int, avg_degree: int) -> int:
    min_num_nodes = max(3 * avg_num_nodes // 4, avg_degree)
    max_num_nodes = 5 * avg_num_nodes // 4
    return random.randint(min_num_nodes, max_num_nodes)

def get_num_nodes_fixed(avg_num_nodes: int):#固定数量的节点
    return avg_num_nodes

def get_num_channels(num_channels) -> int:
    min_num_channels = 3 * num_channels // 4
    max_num_channels = 5 * num_channels // 4
    return random.randint(min_num_channels, max_num_channels)


def get_edge_index(num_src_nodes: int, num_dst_nodes: int, avg_degree: int,
                   is_undirected: bool = False,
                   remove_loops: bool = False) -> torch.Tensor:
    num_edges = num_src_nodes * avg_degree
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.int64)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.int64)
    edge_index = torch.stack([row, col], dim=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index

def get_edge_index_fixed(num_src_nodes: int, num_dst_nodes: int, avg_degree: int,
                   is_undirected: bool = False,
                   remove_loops: bool = False) -> torch.Tensor:
    num_edges = num_src_nodes * avg_degree
    torch.random.manual_seed(10)
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.int64)
    torch.random.seed()
    torch.random.manual_seed(1000)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.int64)
    torch.random.seed()
    edge_index = torch.stack([row, col], dim=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index

def accuary(pred,y):
    correct = (pred== y).sum()
    acc = int(correct) / int(len(y))
    return acc

########################################################################################################################


'''class fakedata'''

class FakeDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool): Whether the graphs to generate are undirected.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)

        data = Data()

        data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)

        if self.num_channels > 0:
            data.x = torch.randn(num_nodes, self.num_channels)
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        return data


class FakeDataset_fixed(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool): Whether the graphs to generate are undirected.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,#改为固定数量的节点数
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        nomalized: bool = False
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.normalized=nomalized

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes_fixed(self.avg_num_nodes)

        data = Data()

        data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)

        if self.num_channels > 0:
            data.x = torch.randn(num_nodes, self.num_channels)
            if self.normalized == True:
                data.x=(data.x-data.x.min())/(data.x.max()-data.x.min())

            #data.x = torch.randn(num_nodes, self.num_channels)
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        return data


class FakeDataset_fixed2(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool): Whether the graphs to generate are undirected.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,#改为固定数量的节点数
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        edgef_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        nomalized: bool = False
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.normalized=nomalized
        self.edgef_dim = edgef_dim
        self.edge_index= get_edge_index_fixed(avg_num_nodes, avg_num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)
        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes_fixed(self.avg_num_nodes)

        data = Data()

        data.edge_index = self.edge_index

        if self.num_channels > 0:
            data.x = torch.randn(num_nodes, self.num_channels)
            if self.normalized == True:
                data.x=(data.x-data.x.min())/(data.x.max()-data.x.min())

            #data.x = torch.randn(num_nodes, self.num_channels)
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
            torch.random.manual_seed(100)
            data.edge_fixed = torch.rand(data.num_edges, self.edgef_dim)
            torch.random.seed()
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        return data


class FakeDataset_fixed_save(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool): Whether the graphs to generate are undirected.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,#改为固定数量的节点数
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        nomalized: bool = False
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.normalized=nomalized

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes_fixed(self.avg_num_nodes)

        data = Data()

        data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)

        if self.num_channels > 0:
            data.x = torch.randn(num_nodes, self.num_channels)
            if self.normalized == True:
                data.x=(data.x-data.x.min())/(data.x.max()-data.x.min())

            #data.x = torch.randn(num_nodes, self.num_channels)
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])


    def process(self):
        torch.save(self.collate([data]), self.processed_paths[0])

'''处理'''
def load_array(data_arrays, batch_size, is_train=True , num_workers=0, pin_memory = False):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers = num_workers, pin_memory = pin_memory)


def process_data_2(data_raw, T, tau):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    avg_num_nodes = data_raw[0].num_nodes
    num_channels = data_raw[0].num_node_features
    num_edges = data_raw[0].edge_attr.shape[0]
    num_edge_fs = data_raw[0].edge_attr.shape[1]
    num_edgef_fs = data_raw[0].edge_fixed.shape[1]
    x = torch.zeros((T, avg_num_nodes, num_channels))
    edge_f = torch.zeros((T, num_edges, num_edge_fs))
    edgefixed_f = torch.zeros((T, num_edges, num_edgef_fs))
    edge_i = torch.zeros((T, 2, num_edges))
    labels = torch.zeros((T - tau, avg_num_nodes, num_channels))

    i = 0
    for data in data_raw:
        x[i] = data.x
        edge_f[i] = data.edge_attr
        edgefixed_f[i] = data.edge_fixed
        edge_i[i] = data.edge_index #恢复edge_index
        if i >= tau:
            j = i - tau
            labels[j] = data.x
        i += 1

    x_features = torch.zeros((T - tau, tau, avg_num_nodes, num_channels))
    edge_features = torch.zeros((T - tau, tau, num_edges, num_edge_fs))
    edgef_features = torch.zeros((T - tau, tau, num_edges, num_edgef_fs))
    edge_index = torch.zeros((T - tau, tau, 2, num_edges))

    for t in range(tau):
        edge_features[:, t] = edge_f[t:T - tau + t]
        edgef_features[:, t] = edgefixed_f[t:T - tau + t]
        x_features[:, t] = x[t:T - tau + t]
        edge_index[:, t] = edge_i[t:T - tau + t]

    return (x_features,edge_features,edgef_features,edge_index,labels)


def process_data(data_raw, T, tau):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    avg_num_nodes = data_raw[0].num_nodes
    num_channels = data_raw[0].num_node_features
    num_edges = data_raw[0].edge_attr.shape[0]
    num_edge_fs = data_raw[0].edge_attr.shape[1]
    num_edgef_fs = data_raw[0].edge_fixed.shape[1]
    x = torch.zeros((T, avg_num_nodes, num_channels))
    edge_f = torch.zeros((T, num_edges, num_edge_fs))
    edgefixed_f = torch.zeros((T, num_edges, num_edgef_fs))
    edge_i = torch.zeros((T, 2, num_edges))
    labels = torch.zeros((T - tau, avg_num_nodes, num_channels))

    i = 0
    for data in data_raw:
        x[i] = data.x
        edge_f[i] = data.edge_attr
        edgefixed_f[i] = data.edge_fixed
        edge_i[i] = data.edge_index  - data.edge_index[0].min()#恢复edge_index
        if i >= tau:
            j = i - tau
            labels[j] = data.x
        i += 1

    x_features = torch.zeros((T - tau, tau, avg_num_nodes, num_channels))
    edge_features = torch.zeros((T - tau, tau, num_edges, num_edge_fs))
    edgef_features = torch.zeros((T - tau, tau, num_edges, num_edgef_fs))
    edge_index = torch.zeros((T - tau, tau, 2, num_edges))

    for t in range(tau):
        edge_features[:, t] = edge_f[t:T - tau + t]
        edgef_features[:, t] = edgefixed_f[t:T - tau + t]
        x_features[:, t] = x[t:T - tau + t]
        edge_index[:, t] = edge_i[t:T - tau + t]

    return [x_features,edge_features,edgef_features,edge_index,labels]

def process_data_mini(data_raw, T, tau,graph_size):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    avg_num_nodes = data_raw[0].num_nodes
    num_channels = data_raw[0].num_node_features
    num_edges = data_raw[0].edge_attr.shape[0]
    num_edge_fs = data_raw[0].edge_attr.shape[1]
    num_edgef_fs = data_raw[0].edge_fixed.shape[1]
    x = torch.zeros((T, avg_num_nodes, num_channels))
    edge_f = torch.zeros((T, num_edges, num_edge_fs))
    edgefixed_f = torch.zeros((T, num_edges, num_edgef_fs))
    edge_i = torch.zeros((T, 2, num_edges))
    labels = torch.zeros((T - tau, avg_num_nodes, num_channels))

    i = 0
    for data in data_raw:
        x[i] = data.x
        edge_f[i] = data.edge_attr
        edgefixed_f[i] = data.edge_fixed
        edge_i[i] = data.edge_index  - data.edge_index[0].min()#恢复edge_index
        if i >= tau:
            j = i - tau
            labels[j] = data.x
        i += 1

    x_features = torch.zeros((T - tau, tau, avg_num_nodes, num_channels))
    edge_features = torch.zeros((T - tau, tau, num_edges, num_edge_fs))
    edgef_features = torch.zeros((T - tau, tau, num_edges, num_edgef_fs))
    edge_index = torch.zeros((T - tau, tau, 2, num_edges))

    for t in range(tau):
        edge_features[:, t] = edge_f[t:T - tau + t]
        edgef_features[:, t] = edgefixed_f[t:T - tau + t]
        x_features[:, t] = x[t:T - tau + t]
        edge_index[:, t] = edge_i[t:T - tau + t]

    return [x_features,edge_features,edgef_features,edge_index,labels]
def generate_traindata(dataset, num_traindata):
    '''
    切分训练集和测试集
    :param dataset:
    :param num_traindata:
    :return:
    '''
    train_data = []
    test_data = []
    for data in dataset:
        train_data.append(data[:num_traindata])
        test_data.append(data[num_traindata:])

    return train_data, test_data

def process_data_normalize(x,edge_attr, edge_fixed,edge_index, T, num_nodes, num_edges,batch_size, num_training, is_edge_flow = False, dataset_type = 'simple', per_test = 0.1, per_train = 0.7, df_split = None):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :param num_training: 训练集图数
    :param dataset_type：数据集划分类型：simple：简单划分为训练和测试；stratified：分层抽样出训练集、验证集和测试集；k-fold：k折；stratified k-fold：分层k折交叉验证
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    import random
    x = x.reshape((T, num_nodes, x.shape[1]))
    for i in range(x.shape[1]):
        x[:,i,:] = (x[:,i,:] - x[:,i,:].min())/(x[:,i,:].max() - x[:,i,:].min())

    if is_edge_flow:
        edge_attr = edge_attr.reshape((T, num_edges, edge_attr.shape[1]))
        for i in range(edge_fixed.shape[1]):
            edge_fixed[:, i] = (edge_fixed[:, i] - edge_fixed[:, i].min()) / (edge_fixed[:, i].max() - edge_fixed[:, i].min())
        for i in range(edge_attr.shape[1]):
            edge_attr[:, i, :] = (edge_attr[:, i, :] - edge_attr[:, i, :].min()) / (edge_attr[:, i, :].max() - edge_attr[:, i, :].min())
    else:
        for i in range(edge_fixed.shape[1]):
            edge_fixed[:,i] = (edge_fixed[:,i]-edge_fixed[:,i].min())/(edge_fixed[:,i].max() - edge_fixed[:,i].min())
            edge_attr[:,i] = edge_fixed[:,i]

        edge_attr[:, edge_attr.shape[1] - edge_fixed.shape[1]-1:] = torch.normal(0.0, 0.01, (edge_attr.shape[0], edge_attr.shape[1] - edge_fixed.shape[1]))
        edge_attr = edge_attr.reshape((T, num_edges, edge_attr.shape[1]))
    edge_fixed = edge_fixed.reshape((T,num_edges,edge_fixed.shape[1]))
    edge_index_new= torch.zeros((T,2, num_edges))
    for i in range(T):
        edge_index_new[i,:,:] = edge_index[:,i*num_edges:(i+1)*num_edges]

    if dataset_type == 'simple':
        dataloader = load_array((x[:num_training], edge_index_new[:num_training], edge_fixed[:num_training], edge_attr[:num_training]), batch_size= batch_size,is_train=False)
        test_dataset = [x[num_training:], edge_index_new[num_training:], edge_fixed[num_training:], edge_attr[num_training:]]
        return dataloader, test_dataset , test_dataset

    elif dataset_type == 'stratified':
        types = len(df_split['type'].drop_duplicates())
        test_x = None
        test_edge_index = None
        test_edge_fixed = None
        test_edge_attr = None

        train_x = None
        train_edge_index = None
        train_edge_fixed = None
        train_edge_attr = None

        val_x = None
        val_edge_index = None
        val_edge_fixed = None
        val_edge_attr = None

        for num_t in range(types):
            list_type = df_split[df_split['type'] == num_t].loc[:,'days'].to_numpy()
            random.shuffle(list_type)
            type_num = len(list_type)
            test_num = int(per_test * type_num)
            train_num = int(per_train * type_num)
            val_num = type_num - test_num -train_num
            for point in range(type_num):
                if point < test_num:
                    index = list_type[point]
                    test_x_p = x[index * 288 : (index + 1) * 288]
                    test_edge_index_p = edge_index_new[index * 288 : (index + 1) * 288]
                    test_edge_fixed_p = edge_fixed[index * 288 : (index + 1) * 288]
                    test_edge_attr_p = edge_attr[index * 288 : (index + 1) * 288]
                    if test_x == None:
                        test_x = test_x_p
                        test_edge_index = test_edge_index_p
                        test_edge_fixed = test_edge_fixed_p
                        test_edge_attr = test_edge_attr_p
                    else:
                        test_x = torch.cat((test_x, test_x_p),dim=0)
                        test_edge_index = torch.cat((test_edge_index, test_edge_index_p), dim=0)
                        test_edge_fixed = torch.cat((test_edge_fixed, test_edge_fixed_p), dim=0)
                        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr_p), dim=0)

                elif test_num <=point < test_num + train_num:
                    index = list_type[point]
                    train_x_p = x[index * 288: (index + 1) * 288]
                    train_edge_index_p = edge_index_new[index * 288: (index + 1) * 288]
                    train_edge_fixed_p = edge_fixed[index * 288: (index + 1) * 288]
                    train_edge_attr_p = edge_attr[index * 288: (index + 1) * 288]
                    if train_x == None:
                        train_x = train_x_p
                        train_edge_index = train_edge_index_p
                        train_edge_fixed = train_edge_fixed_p
                        train_edge_attr = train_edge_attr_p
                    else:
                        train_x = torch.cat((train_x, train_x_p), dim=0)
                        train_edge_index = torch.cat((train_edge_index, train_edge_index_p), dim=0)
                        train_edge_fixed = torch.cat((train_edge_fixed, train_edge_fixed_p), dim=0)
                        train_edge_attr = torch.cat((train_edge_attr, train_edge_attr_p), dim=0)

                else:
                    index = list_type[point]
                    val_x_p = x[index * 288: (index + 1) * 288]
                    val_edge_index_p = edge_index_new[index * 288: (index + 1) * 288]
                    val_edge_fixed_p = edge_fixed[index * 288: (index + 1) * 288]
                    val_edge_attr_p = edge_attr[index * 288: (index + 1) * 288]
                    if val_x == None:
                        val_x = val_x_p
                        val_edge_index = val_edge_index_p
                        val_edge_fixed = val_edge_fixed_p
                        val_edge_attr = val_edge_attr_p
                    else:
                        val_x = torch.cat((val_x, val_x_p), dim=0)
                        val_edge_index = torch.cat((val_edge_index, val_edge_index_p), dim=0)
                        val_edge_fixed = torch.cat((val_edge_fixed, val_edge_fixed_p), dim=0)
                        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr_p), dim=0)
        #重置索引顺序，以免报错
        index_0 = edge_index_new[0]
        #首先对train_index进行重排
        for i in range(train_edge_index.shape[0]):
            train_edge_index[i] = index_0 + i * num_nodes

        for j in range(val_edge_index.shape[0]):
            val_edge_index[j] = index_0 + (i + j + 1) * num_nodes

        for k in range(test_edge_index.shape[0]):
            test_edge_index[k] = index_0 + ( k +i +j +2) * num_nodes

        dataloader = load_array((train_x , train_edge_index, train_edge_fixed, train_edge_attr), batch_size = batch_size, is_train=False)
        val_dataset = [val_x , val_edge_index, val_edge_fixed, val_edge_attr]
        test_dataset = [test_x , test_edge_index, test_edge_fixed, test_edge_attr]
        return dataloader, val_dataset, test_dataset



def process_data_x_GRU( x_raw, T, tau,pred_steps, num_nodes):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    num_channels = x_raw.shape[2]
    x_features = torch.zeros((T - tau - pred_steps +1 , tau, num_nodes, num_channels))

    for t in range(tau):
        x_features[:, t] = x_raw[t:T - tau + t -pred_steps +1]

    return x_features



def process_data_normalize_GRU(x,edge_fixed,edge_attr, T, tau, pred_steps, num_nodes, num_edges,batch_size, num_training,device):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :param num_training: 训练集图数
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    x = x.reshape((T, num_nodes, x.shape[1]))
    for i in range(edge_fixed.shape[1]):
        edge_fixed[:, i] = (edge_fixed[:, i] - edge_fixed[:, i].min()) / (edge_fixed[:, i].max() - edge_fixed[:, i].min())
        edge_attr[:, i] = edge_fixed[:, i]

    edge_attr[:, edge_attr.shape[1] - edge_fixed.shape[1] - 1:] = torch.normal(0.0, 0.01, (edge_attr.shape[0], edge_attr.shape[1] - edge_fixed.shape[1]))
    edge_fixed = edge_fixed.reshape((T, num_edges, edge_fixed.shape[1]))
    edge_attr = edge_attr.reshape((T, num_edges, edge_attr.shape[1]))

    for i in range(x.shape[1]):
        x[:,i,:] = (x[:,i,:] - x[:,i,:].min())/(x[:,i,:].max() - x[:,i,:].min())


    labels = process_data_labels_3ds(x_0 = x[:num_training], T = num_training, tau= tau, pred_steps= pred_steps,num_nodes= num_nodes,num_edges= num_edges,device=device)
    x_new = process_data_x_GRU(x_raw= x[:num_training], T = num_training, tau = tau, num_nodes = num_nodes, pred_steps= pred_steps)
    edge_attr_new = process_data_x_GRU(x_raw= edge_attr[:num_training], T = num_training, tau = tau, num_nodes = num_edges, pred_steps= pred_steps)
    test_labels = process_data_labels_3ds(x[num_training:], T = T - num_training, tau =tau, pred_steps=pred_steps, num_nodes= num_nodes, num_edges= num_edges,device=device)
    test_x = process_data_x_GRU(x_raw= x[num_training:], T = T - num_training, tau = tau, num_nodes = num_nodes,pred_steps= pred_steps)
    test_edge_attr = process_data_x_GRU(x_raw= edge_attr[num_training:], T = T - num_training, tau = tau, num_nodes = num_edges, pred_steps= pred_steps)
    test_dataset = [test_x,test_edge_attr,test_labels]
    dataloader = load_array((x_new, edge_attr_new, labels), batch_size= batch_size,is_train=False)
    return dataloader, test_dataset


'''处理为序列训练数据'''
def process_data_sequence(x_0, x_raw,edge_attr_raw, T, tau, num_nodes, num_edges,device):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    num_channels = x_raw.shape[1]
    num_edge_fs = edge_attr_raw.shape[1]
    x = x_raw.reshape((T,num_nodes,num_channels))
    edge_attr = edge_attr_raw.reshape((T,num_edges,num_edge_fs))
    labels = x_0[(tau* num_nodes):,:]
    labels = labels.reshape((T - tau, num_nodes, x_0.shape[1]))

    x_features = torch.zeros((T - tau, tau, num_nodes, num_channels),device=device)
    edge_features = torch.zeros((T - tau, tau, num_edges, num_edge_fs),device=device)

    for t in range(tau):
        edge_features[:, t] = edge_attr[t:T - tau + t]
        x_features[:, t] = x[t:T - tau + t]

    return (x_features,edge_features,labels)

'''处理为序列训练数据'''
def process_data_x( x_raw,edge_attr_raw, T, tau, pred_steps, num_nodes, num_edges,device):
    #读取数据
    '''

    :param data_raw: 未处理数据
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    num_channels = x_raw.shape[1]
    num_edge_fs = edge_attr_raw.shape[1]
    batch = int(x_raw.shape[0] / num_nodes)
    x = x_raw.reshape((batch,num_nodes,num_channels))
    edge_attr = edge_attr_raw.reshape((batch,num_edges,num_edge_fs))

    x_features = torch.zeros((batch - tau - pred_steps + 1, tau, num_nodes, num_channels),device=device)
    edge_features = torch.zeros((batch - tau - pred_steps + 1, tau, num_edges, num_edge_fs),device=device)

    for t in range(tau):
        edge_features[:, t] = edge_attr[t:batch - tau + t - pred_steps + 1]
        x_features[:, t] = x[t:batch - tau + t - pred_steps + 1]

    return (x_features,edge_features)

def process_data_labels(x_0, T, tau, pred_steps, num_nodes, num_edges, device):
    #读取数据
    '''

    :param data_raw: 未处理数据[T*num_nodes,x_channels]
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据[T,num_nodes, x_channels]
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    batch = int(x_0.shape[0]/num_nodes)
    labels = torch.zeros((batch - tau -pred_steps + 1, pred_steps  , num_nodes, x_0.shape[-1]), device = device)
    x_0 = x_0.reshape((batch , num_nodes, x_0.shape[-1]))
    for t in range(pred_steps):
        labels[:, t, :, :] = x_0[tau+t : batch - pred_steps + t + 1 ]

    return labels

def process_data_labels_3ds(x_0, T, tau,pred_steps, num_nodes, num_edges,device):
    #读取数据
    '''

    :param data_raw: 未处理数据[T, num_nodes,x_channels]
    :param batch_size1: 一张一张图处理
    :param T: 总图数
    :param tau: 步长
    :return: 处理后数据[T,num_nodes, x_channels]
    '''
    '''
       avg_num_nodes = data_raw[0].num_nodes：节点数量
       num_channels = data_raw[0].num_node_features：节点特征数
       num_edges = data_raw[0].edge_attr.shape[0]：边数量
       num_edge_fs = data_raw[0].edge_attr.shape[1]：边变动特征数
       num_edgef_fs = data_raw[0].edge_fixed.shape[1]：边不变特征数
       loader = DataLoader(data_raw, batch_size1, shuffle=False)
       x = torch.zeros((T, avg_num_nodes, num_channels))
       edge_f = torch.zeros((T, e.shape[0], e.shape[1]))
       edgefixed_f = torch.zeros((T, e.shape[0], e.shape[1]))
       labels = torch.zeros((T - tau, avg_num_nodes, tau))
       '''
    #labels = x_0[tau:,:,:]
    labels = torch.zeros((T - tau - pred_steps + 1, pred_steps, num_nodes, x_0.shape[-1]), device=device)
    x_0 = x_0.reshape((T, num_nodes, x_0.shape[-1]))
    for t in range(pred_steps):
        labels[:, t, :, :] = x_0[tau + t: T - pred_steps + t + 1]

    return labels



class CudaDataLoader:
    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ Append the data into the queue in a while-true loop"""
        # The loop that will load into the queue in the background
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ Load the data to specified device """
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) in (list, str):
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
    """ 一直repeat的sampler """



'''class model'''
class EdgeConvf(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, node_in , Edge_in , Edgef_in, Edge_out,aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.edge_new = None
        self.lin_ec = nn.Linear(node_in +  Edge_in + Edgef_in, Edge_out)
        self.lin_for_node = nn.Linear(node_in, node_in)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.lin_ec)
        reset(self.lin_for_node)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Optional, edge_fixed: Optional) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        self.propagate(edge_index, x=x, edge_attr= edge_attr, edge_fixed=edge_fixed,size=None)
        return self.edge_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional, edge_fixed: Optional) -> Tensor:
        message_ = torch.cat(( self.lin_for_node(x_j - x_i), edge_attr,edge_fixed), dim = -1)
        message_ = self.lin_ec(message_)
        self.edge_new = message_
        return message_

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.lin_ec})'


class Edgeconvfc(nn.Module):
    def __init__(self,E_in,E_hiddens,N_in,N_hiddens,Ef,**kwargs):
        # super().__init__()
        super(Edgeconvfc, self).__init__(**kwargs)
        self.E_hiddens = E_hiddens#输出的边特征数
        self.N_in = N_in#节点特征数
        self.E_in = E_in#输入边特征数
        self.N_hiddens = N_hiddens#节点隐藏特征数
        self.Ef=Ef#固定边特征数
        self.in_channels=self.N_hiddens+self.E_in+self.Ef#输入第二个线性层：节点隐藏+边输入+固定边特征
        self.lin1=nn.Linear(in_features=self.N_in,out_features=self.N_hiddens)
        self.lin2=nn.Linear(in_features=self.in_channels,out_features=self.E_hiddens)
    def forward(self,x,edge_index,edge_f,edge_attr,device):
        out=torch.zeros((edge_index.shape[1],self.E_hiddens),device= device)
        start_ID =edge_index[0].min()#确定最小的节点编号
        for i in range(edge_index.shape[1]):
            x_j=x[edge_index[0,i] - start_ID] #减去起始ID，使x的索引能与i对应上
            x_i=x[edge_index[1,i] - start_ID]
            x_em=self.lin1(x_j)-self.lin1(x_i)
            x_em=F.relu(x_em)
            in_features=torch.cat((x_em,edge_attr[i],edge_f[i]),0)
            edge_features=self.lin2(in_features)
            out[i]=edge_features
        return out

class Edgeconvf(nn.Module):
    def __init__(self,E_in,E_hiddens,N_in,N_hiddens,Ef,**kwargs):
        # super().__init__()
        super(Edgeconvf, self).__init__(**kwargs)
        self.E_hiddens = E_hiddens#输出的边特征数
        self.N_in = N_in#节点特征数
        self.E_in = E_in#输入边特征数
        self.N_hiddens = N_hiddens#节点隐藏特征数
        self.Ef=Ef#固定边特征数
        self.in_channels=self.N_hiddens+self.E_in+self.Ef#输入第二个线性层：节点隐藏+边输入+固定边特征
        self.lin1=nn.Linear(in_features=self.N_in,out_features=self.N_hiddens)
        self.lin2=nn.Linear(in_features=self.in_channels,out_features=self.E_hiddens)
    def forward(self,x,edge_index,edge_f,edge_attr,device):
        out=torch.zeros((edge_index.shape[1],self.E_hiddens),device= device)
        #start_ID = edge_index[0].min()
        for i in range(edge_index.shape[1]):
            #x_j=x[edge_index[0,i]-start_ID]
            #x_i=x[edge_index[1,i]- start_ID]
            x_j = x[edge_index[0, i]]
            x_i = x[edge_index[1, i]]
            x_em=self.lin1(x_j)+self.lin1(x_i)
            x_em=F.relu(x_em)
            in_features=torch.cat((x_em,edge_attr[i],edge_f[i]),0)
            edge_features=self.lin2(in_features)
            out[i]=edge_features
        return out


class Res_EGblk_0(nn.Module):
    def __init__(self, edge_in, edge_out, node_in, node_out, Ef, use_nodelin=False, use_edgelin=False, **kwargs):
        # super().__init__()
        super(Res_EGblk, self).__init__(**kwargs)
        self.node_in =node_in
        self.node_out =node_out
        self.edge_in = edge_in
        self.edge_out = edge_out
        self.gcn = pyg_nn.GCNConv(in_channels=node_in, out_channels=node_out)
        self.edg = Edgeconvf(E_in = edge_in,E_hiddens = edge_out, N_in = node_in, N_hiddens = node_in,Ef=Ef)
        self.linw=nn.Linear(in_features= edge_out, out_features= 1)
        self.relu = nn.ReLU()
        self.bn_e = nn.BatchNorm1d(edge_out)
        self.bn_n = nn.BatchNorm1d(node_out)
        if use_nodelin:
            self.lin4_node = nn.Linear(in_features=node_in, out_features=node_out)
        else:
            self.lin4_node = None

        if use_edgelin:
            self.lin4_edge = nn.Linear(in_features=edge_in, out_features=edge_out)

        else:
            self.lin4_edge = None

    def forward(self,data):
        #print(self.node_in,self.node_out,self.edge_in, self.edge_out)
        x, edge_index, edge_f, edge_attr, device= data[0], data[1],data[2],data[3],data[4]
        edge_features = self.edg(x, edge_index, edge_f, edge_attr,device)
        edge_features = self.bn_e(edge_features)
        edge_features = self.relu(edge_features)
        edge_weight = self.relu((self.linw(edge_features)))
        x_1 = self.gcn(x, edge_index,edge_weight)
        x_1 = self.bn_n(x_1)
        x_1 = self.relu(x_1)
        if self.lin4_node:
            x = self.lin4_node(x)

        if self.lin4_edge:
            edge_attr = self.lin4_edge(edge_attr)


        return (x_1 + x, edge_index, edge_f, edge_features + edge_attr,device)

class Res_EGblk_1(nn.Module):
    def __init__(self, edge_in, edge_out, node_in, node_out, Ef, use_nodelin=False, use_edgelin=False, **kwargs):
        # super().__init__()
        super(Res_EGblk, self).__init__(**kwargs)
        self.node_in =node_in
        self.node_out =node_out
        self.edge_in = edge_in
        self.edge_out = edge_out
        self.edf = egf.EGCNConv(Edge_in= edge_in, Edge_out= edge_out, Edgef_in= Ef, in_channels= node_in, out_channels= node_out, add_self_loops= False)
        self.relu = nn.ReLU()
        self.bn_e = nn.BatchNorm1d(edge_out)
        self.bn_n = nn.BatchNorm1d(node_out)
        if use_nodelin:
            self.lin4_node = nn.Linear(in_features=node_in, out_features=node_out)
        else:
            self.lin4_node = None

        if use_edgelin:
            self.lin4_edge = nn.Linear(in_features=edge_in, out_features=edge_out)

        else:
            self.lin4_edge = None

    def forward(self,data):
        #print(self.node_in,self.node_out,self.edge_in, self.edge_out)
        x, edge_index, edge_f, edge_attr, device= data[0], data[1],data[2],data[3],data[4]
        x_1, edge_features = self.edf(x = x, edge_index = edge_index, edge_attr = edge_attr, edge_fixed = edge_f)
        edge_features = self.bn_e(edge_features)
        edge_features = self.relu(edge_features)
        x_1 = self.bn_n(x_1)
        x_1 = self.relu(x_1)
        if self.lin4_node:
            x = self.lin4_node(x)

        if self.lin4_edge:
            edge_attr = self.lin4_edge(edge_attr)


        return (x_1 + x, edge_index, edge_f, edge_features + edge_attr,device)

class Res_EGblk(nn.Module):
    def __init__(self, edge_in, edge_out, node_in, node_out, Ef, use_nodelin=False, use_edgelin=False, **kwargs):
        # super().__init__()
        super(Res_EGblk, self).__init__(**kwargs)
        self.node_in =node_in
        self.node_out =node_out
        self.edge_in = edge_in
        self.edge_out = edge_out
        self.gcn = pyg_nn.GCNConv(in_channels=node_in, out_channels=node_out)
        self.edg = EdgeConvf(Edge_in= edge_in, Edge_out= edge_out, Edgef_in= Ef, node_in= node_in)
        self.linw=nn.Linear(in_features= edge_out, out_features= 1)
        self.relu = nn.ReLU()
        #self.bn_e = nn.BatchNorm1d(edge_out)
        #self.bn_n = nn.BatchNorm1d(node_out)
        if use_nodelin:
            self.lin4_node = nn.Linear(in_features=node_in, out_features=node_out)
        else:
            self.lin4_node = None

        if use_edgelin:
            self.lin4_edge = nn.Linear(in_features=edge_in, out_features=edge_out)

        else:
            self.lin4_edge = None

    def forward(self,data):
        #print(self.node_in,self.node_out,self.edge_in, self.edge_out)
        x, edge_index, edge_f, edge_attr, device= data[0], data[1],data[2],data[3],data[4]
        edge_features = self.edg(x = x, edge_index = edge_index, edge_fixed = edge_f, edge_attr = edge_attr)
        #edge_features = self.bn_e(edge_features)
        edge_features = self.relu(edge_features)
        edge_weight = self.relu((self.linw(edge_features)))
        x_1 = self.gcn(x, edge_index, edge_weight)
        #x_1 = self.bn_n(x_1)
        x_1 = self.relu(x_1)
        if self.lin4_node:
            x = self.lin4_node(x)

        if self.lin4_edge:
            edge_attr = self.lin4_edge(edge_attr)


        return (x_1 + x, edge_index, edge_f, edge_features + edge_attr,device)


def Resegnet_block(edge_in, edge_out, node_in, node_out, Ef, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Res_EGblk(edge_in=edge_in, edge_out =edge_out, node_in= node_in, node_out=node_out, Ef=Ef,
                                use_nodelin=True, use_edgelin=True))
        else:
            blk.append(Res_EGblk(edge_in=edge_out, edge_out=edge_out, node_in=node_out, node_out=node_out, Ef=Ef))
    return blk


class ResEG_GRU(nn.Module):
    def __init__(self,net_ResEG,**kwargs):
        super(ResEG_GRU, self).__init__(**kwargs)
        self.net_ResEG = net_ResEG
        self.en_gru = nn.GRU(input_size=5376, hidden_size= 800,batch_first= True, dropout= 0.15)
        #self.de_gru = nn.GRU(input_size=6400, hidden_size= 800, batch_first = True, dropout= 0.15)
        self.mlp = nn.Sequential(nn.Linear(in_features= 5* 800, out_features= 1280),nn.Linear(in_features= 1280, out_features= 5*80))

    def forward(self, x, edge_index, edge_f, edge_attr):
        x_1=torch.zeros((x.shape[0],x.shape[1],x.shape[2],8),device= device)
        e_1 =torch.zeros((edge_attr.shape[0],edge_attr.shape[1],edge_attr.shape[2],8),device= device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                edge_i = edge_index[i,j] - edge_index[i,j,0].min()
                data= self.net_ResEG((x[i,j], edge_i, edge_f[i,j], edge_attr[i,j],device))
                x_1[i, j], e_1[i, j]  = data[0], data[3]
        x_1 = x_1.reshape((x_1.shape[0], x_1.shape[1], x_1.shape[2] * x_1.shape[3]))
        e_1 = e_1.reshape((e_1.shape[0], e_1.shape[1], e_1.shape[2] * e_1.shape[3]))
        y_hat = torch.cat((x_1, e_1),dim=2)
        y_hat,_ = self.en_gru(y_hat)
        #y_hat,_ = self.de_gru(y_hat)
        y_hat = y_hat.reshape((y_hat.shape[0],y_hat.shape[1]*y_hat.shape[2]))
        y_hat = self.mlp(y_hat)
        y_hat = y_hat.reshape((y_hat.shape[0], 80, 5))
        return y_hat

'''training'''

def train_for_graph(epochs,optimizer,dataloader,model,device,loss):#for graph
    for epoch in range(epochs):
        for data in dataloader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            pred = out.argmax(dim=1)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            l=loss(out,data.y)
            l.mean().backward()
            optimizer.step()
        acc=accuary(pred,data.y)
        print(f'epoch{epoch + 1},'f'loss:{l.sum() / len(data.y):f}','acc:',acc)


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr



