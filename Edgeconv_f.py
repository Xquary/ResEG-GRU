import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from torch_geometric.nn.inits import reset

try:
    from torch_cluster import knn
except ImportError:
    knn = None

from typing import Optional, Tuple,Union

import torch
from torch import Tensor
from torch.nn import Parameter,ReLU
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class EGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 Edge_in: int, Edge_out :int,
                 Edgef_in: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.relu = ReLU()
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin_for_edge = Linear(Edge_out, 1)
        self.lin_ec = Linear( out_channels+  Edge_in + Edgef_in, Edge_out)
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.edge_new = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, edge_fixed: OptTensor = None, edge_weight: Optional = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             edge_fixed = edge_fixed,edge_weight = edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out, self.edge_new

    def message(self,x_i: Tensor, x_j: Tensor, edge_weight: OptTensor, edge_attr: Optional, edge_fixed: Optional) -> Tensor:
        message_ = torch.cat(( x_j - x_i, edge_attr,edge_fixed), dim = -1)
        message_ = self.lin_ec(message_)
        self.edge_new = message_
        edge_weight = self.lin_for_edge(message_)
        edge_weight = self.relu(edge_weight)
        return x_j if edge_attr is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

'''
params={'E_hiddens':64, 'N_in': 5, 'E_in': 5, 'N_hiddens': 10, 'Ef': 4}
class Edgeconvf(nn.Module):
    def __init__(self, params,**kwargs):
        # super().__init__()
        super(Edgeconvf, self).__init__(**kwargs)
        self.E_hiddens=params['E_hiddens']#输出的边特征数
        self.N_in=params['N_in']#节点特征数
        self.E_in=params['E_in']#输入边特征数
        self.N_hiddens=params['N_hiddens']#节点隐藏特征数
        self.Ef=params['Ef']#固定边特征数
        self.in_channels=self.N_hiddens+self.E_in+self.Ef#输入第二个线性层：节点隐藏+边输入+固定边特征
        self.lin1=nn.Linear(in_features=self.N_in,out_features=self.N_hiddens)
        self.lin2=nn.Linear(in_features=self.in_channels,out_features=self.E_hiddens)
    def forward(self,x,edge_index,edge_f,edge_attr):
        out=torch.zeros((edge_index.shape[1],self.E_hiddens))
        for i in range(edge_index.shape[1]):
            x_j=x[edge_index[0,i]]
            x_i=x[edge_index[1,i]]
            x_em=self.lin1(x_j)-self.lin1(x_i)
            x_em=F.relu(x_em)
            in_features=torch.cat((x_em,edge_attr[i],edge_f[i]),0)
            edge_features=self.lin2(in_features)
            out[i]=edge_features
        return out


class Res_EGblk(nn.Module):
    def __init__(self, edge_in, edge_out, node_in, node_out, use_nodelin = False, use_edgelin = False, **kwargs):
        # super().__init__()
        super(Res_EGblk, self).__init__(**kwargs)
        self.gcn = pyg_nn.GCNConv(in_channels=node_in ,out_channels=node_out)
        self.edg = Edgeconvf(params)
        if use_nodelin:
            self.lin4_node = nn.Linear(in_features=node_in, out_features = node_out)
        else:
            self.lin4_node = None

        if use_edgelin:
            self.lin4_edge = nn.Linear(in_features=edge_in, out_features=edge_out)

        else:
            self.lin4_edge = None

    def forward(self,x,edge_index,edge_f,edge_attr):
        edge_features = self.edg(x,edge_index,edge_f,edge_attr)
        edge_features = self.relu(edge_features)
        x_1 = self.gcn(x,edge_index,edge_features)
        x_1 = F.relu(x)
        if self.lin4_node:
            x = self.lin4_node(x)

        if self.lin4_edge:
            edge_attr = self.lin4_edge(edge_attr)

        return x_1+x, edge_features+edge_attr

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


edf=Edgeconvf(params)
o=edf(x,edge_index,edge_f,edge_attr)
'''

