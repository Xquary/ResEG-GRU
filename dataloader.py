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
import gcn_basic_pack as gbp


def createfakegraph(num_graphs: int = 1,
        avg_num_nodes: int = 1000,#改为固定数量的节点数
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        edgef_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform = None,
        pre_transform = None,
        nomalized: bool = False,
        save: bool = False,
        root = '3.pt'):
    dataset=gbp.FakeDataset_fixed2(num_graphs, avg_num_nodes, avg_degree, num_channels, edge_dim, edgef_dim, num_classes, task, is_undirected, transform,
                                  pre_transform)
    if save:
        saved=colt.collate(dataset[0].__class__, dataset)
        torch.save(saved, root)
        return dataset,saved
    else:
        return dataset


class Getdata(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Getdata, self).__init__(root, transform, pre_transform)
        a=torch.load(self.processed_paths[0])
        self.data, self.slices = a[0] ,a[1]
    '''
    def processed_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]
    '''
    @property
    def processed_file_names(self) :
        return '2.pt'
        #return self.file_names

#a=Getdata('D:\py4fun\pyg\MyModel')
d=createfakegraph(num_graphs=20,avg_num_nodes=80,avg_degree=4,num_channels=5, num_classes=10,edge_dim=5,edgef_dim=5, nomalized=True,save=True , root= '4.pt')
