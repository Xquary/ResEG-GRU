from typing import Optional, Callable

import numpy as np
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
from gcn_basic_pack import Resegnet_block
from torch_geometric.loader import DataLoader
from d2l import torch as d2l
from torch.optim import lr_scheduler
import time
import fastdataloader as fdl
from torch.utils import data
import math
import model_pack as mp
from matplotlib import pyplot as plt


'''数据读取类'''


class Getdata(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Getdata, self).__init__(root, transform, pre_transform)
        a = torch.load(self.processed_paths[0])
        self.data, self.slices = a[0], a[1]

    '''
    def processed_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]
    '''

    @property
    def processed_file_names(self):
        return '2880_graphs_36_192.pt'

#%%
Data_raw = Getdata('D:\py4fun\pyg\MyModel')

"""训练集"""
x=Data_raw.data.x
edge_attr = Data_raw.data.edge_attr
edge_fixed = Data_raw.data.edge_fixed
edge_index = Data_raw.data.edge_index

'''处理为序列训练数据'''
def process_data_x_GRU( x_raw, T, tau, num_nodes):
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
    x_features = torch.zeros((T - tau, tau, num_nodes, num_channels))

    for t in range(tau):
        x_features[:, t] = x_raw[t:T - tau + t]

    return x_features

"""测试集"""

def process_data_normalize_GRU(x,edge_fixed,edge_attr, T, tau, num_nodes, num_edges,batch_size, num_training):
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


    labels = gbp.process_data_labels_3ds(x_0 = x[:num_training], T = num_training, tau= tau, num_nodes= num_nodes,num_edges= num_edges)
    x_new = process_data_x_GRU(x_raw= x[:num_training], T = num_training, tau = tau, num_nodes = num_nodes)
    edge_attr_new = process_data_x_GRU(x_raw= edge_attr[:num_training], T = num_training, tau = tau, num_nodes = num_edges)
    test_labels = gbp.process_data_labels_3ds(x[num_training:], T = num_training, tau =5, num_nodes= num_nodes, num_edges= num_edges)
    test_x = process_data_x_GRU(x_raw= x[num_training:], T = T - num_training, tau = tau, num_nodes = num_nodes)
    test_edge_attr = process_data_x_GRU(x_raw= edge_attr[num_training:], T = T - num_training, tau = tau, num_nodes = num_edges)
    test_dataset = [test_x,test_edge_attr,test_labels]
    dataloader = gbp.load_array((x_new, edge_attr_new, labels), batch_size= batch_size,is_train=False)
    return dataloader, test_dataset





#%%
dataloader,test_dataset = process_data_normalize_GRU(x = x, edge_attr= edge_attr, edge_fixed= edge_fixed,tau=5, T=2880, num_nodes=36, num_edges=192, batch_size=256, num_training= 2016)
#%%
def rc_edge_index(edge_index,device):
    """

    :param edge_index: [batch , 2, num_edges_for_1_graph]
    :return: [2,batch * num_edges_for_1_graph]
    """
    edge_index_rc = torch.zeros((edge_index.shape[1], edge_index.shape[0]*edge_index.shape[2]),device= device)
    for i in range(edge_index.shape[0]):
        edge_index_rc[:, edge_index.shape[2]*i:edge_index.shape[2]*(i+1)] = edge_index[i]
    edge_index_rc = edge_index_rc - edge_index_rc[0].min()
    return edge_index_rc

'''梯度裁剪'''
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
#%%

def train_for_GRU_Cudaloader(epochs,optimizer, dataloader, test_dataset,  model, device, T, tau, batch_size, loss, num_nodes, num_edges, scheduler= None):#for graph
    train_loss_collections=[]
    val_loss_collections = []
    x_val = test_dataset[0].to(device)
    edge_attr_val = test_dataset[1].to(device)
    y_val = test_dataset[2].to(device)
    x_val = x_val.reshape((x_val.shape[0],x_val.shape[1],x_val.shape[2]*x_val.shape[3]))
    edge_attr_val = edge_attr_val.reshape((edge_attr_val.shape[0], edge_attr_val.shape[1], edge_attr_val.shape[2] * edge_attr_val.shape[3]))
    #embedding_val = torch.cat((x_val, edge_attr_val),dim= 2)
    embedding_val = x_val
    start_time = time.time()
    for epoch in range(epochs):
        batch = 0
        for x, edge_attr, labels in dataloader:
            time_pin1 = time.time()
            x = x.reshape((x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
            edge_attr = edge_attr.reshape((edge_attr.shape[0], edge_attr.shape[1], edge_attr.shape[2] * edge_attr.shape[3]))
            #embedding = torch.cat((x, edge_attr),dim= 2)
            embedding = x
            y_hat = model(embedding)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            l=loss(y_hat,labels)
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.zero_grad()
                l.mean().backward()
                grad_clipping(model, 1)
                optimizer.step()
            else:
                l.mean().backward()
                grad_clipping(model, 1)
                # 因为已经调用了mean函数
                optimizer.step()
            time_pin2 = time.time()
            batch_time = round(time_pin2 - time_pin1,2)
            print(f'epoch{epoch + 1},'f'batch{batch + 1}'f' of {math.ceil((T-tau)/batch_size)}',f'process time:{batch_time}'f's')
            batch += 1

        y_hat_val = model(embedding_val)
        val_loss = loss(y_hat_val, y_val)

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # UsingPyTorchIn-Builtscheduler
                scheduler.step(l.mean())
            else:
                # Usingcustomdefinedscheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

        #acc=accuary(pred,data.y)
        #print(f'epoch{epoch + 1},'f'loss:{l.sum() / len(data.y):f}','acc:',acc)
        #loss_collections.append(l.sum() / y.numel())
        print(f'epoch{epoch + 1},'f'train_loss:{l.sum() / labels.numel():f}',f'val_loss:{val_loss.sum() / y_val.numel():f}')
        train_loss_collections.append(l.mean().cpu().detach().numpy())
        val_loss_collections.append(val_loss.mean().cpu().detach().numpy())
    end_time = time.time()
    total_time = round(end_time - start_time,2)
    print('total time:',total_time,'s')
    return train_loss_collections, val_loss_collections

#%%
T_train= 2016
tau = 5
'''model'''
"""Basic Two_layer GRU"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= mp.EncoderDecoder_GRU().to(device)
#model = mp.EncoderDecoder_LSTM().to(device)
updater = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.MSELoss(reduction='none')
#scheduler= gbp.CosineScheduler(max_update=10, base_lr=0.001, final_lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(mode= 'min', factor=0.1,verbose=True,optimizer= updater)
if torch.cuda.is_available():
    dataloader = fdl.CudaDataLoader(dataloader, device=0)
#%%
#train_for_Res(2,optimizer = updater, dataloader = dataloader, model = model, device = device, T = T, tau = tau, batch_size =32, loss = loss, num_nodes = 36, num_edges =194, scheduler= None)
train_loss_collections, val_loss_collections=train_for_GRU_Cudaloader(200, optimizer = updater, dataloader = dataloader,
                                                                      test_dataset= test_dataset, model = model,
                                                                      device = device, T = T_train, tau = tau, batch_size =256,
                                                                      loss = loss, num_nodes = 36, num_edges =192,
                                                                      scheduler= scheduler)

