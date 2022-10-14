from typing import Optional, Callable
import pandas as pd
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
import training_pack as tp
from matplotlib import pyplot as plt
import os


class Getdata0(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Getdata0, self).__init__(root, transform, pre_transform)
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
        return '384_graphs_36_186.pt'
class Getdata1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Getdata1, self).__init__(root, transform, pre_transform)
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
        return '2880_graphs_72_656.pt'


class Getdata2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Getdata2, self).__init__(root, transform, pre_transform)
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
        return '16992_graphs_69_468.pt'

'''数据读取类'''
def getdata(num, root):
    if num == 0:
        data = Getdata0(root)
        T = 384
        T_train = 288
        batch_size = 32
        num_nodes = 36
        num_edges = 186
    if num == 1:
        data = Getdata1(root)
        T = 2880
        T_train = 2016
        batch_size = 256
        num_nodes = 72
        num_edges = 656

    if num == 2:
        data = Getdata2(root)
        T = 16992
        T_train = 11520
        batch_size = 288
        num_nodes = 69
        num_edges = 468

    return data, T, T_train, batch_size, num_nodes, num_edges


def Experiment(data_list, root, tau_list, pred_list, num_exp,df_split, result = False):
    print('running...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dt in data_list:
        Data_raw, T, T_train, batch_size,num_nodes, num_edges = getdata(dt, root)
        x = Data_raw.data.x.to(torch.float32)
        edge_attr = Data_raw.data.edge_attr
        edge_fixed = Data_raw.data.edge_fixed
        edge_index = Data_raw.data.edge_index

        for tau in tau_list:
            df_em = pd.DataFrame(columns=['model', 'step', 'pred', 'mse', 'mae', 'mape', 'accuracy', 'r2', 'var'])
            for pred_steps in pred_list:
                for exp in range(num_exp):
                    days_step = 2
                    num_training = T_train + days_step * 288 * exp
                    dataloader, val_dataset, test_dataset = gbp.process_data_normalize(x, edge_attr, edge_fixed,
                                                                                       edge_index, T=T,
                                                                                       num_nodes=num_nodes,
                                                                                       num_edges=num_edges,
                                                                                       batch_size=batch_size,
                                                                                       num_training=num_training,
                                                                                       dataset_type='simple',
                                                                                       df_split=df_split)
                    if torch.cuda.is_available():
                        dataloader = fdl.CudaDataLoader(dataloader, device=0)
                    b1 = nn.Sequential(
                        *Resegnet_block(edge_in=5, edge_out=5, node_in=5, node_out=5, Ef=2, num_residuals=1,
                                        first_block=True))
                    b2 = nn.Sequential(*Resegnet_block(edge_in=5, edge_out=8, node_in=5, node_out=8, Ef=2, num_residuals=3))
                    net_ResEG = nn.Sequential(b1, b2).to(device)
                    model = mp.ResEG_DE_Model(net_ResEG, tau, pred_steps = pred_steps, num_nodes= num_nodes).to(device)
                    updater = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss = nn.MSELoss(reduction='none')
                    scheduler = lr_scheduler.ReduceLROnPlateau(mode='min', factor=0.1, verbose=True, optimizer=updater)
                    train_loss_collections, val_loss_collections, epoch_time , ev_list, y_hat_test = tp.train_for_Res_Cudaloader(100, optimizer=updater,
                                                                                               dataloader=dataloader,
                                                                                                val_dataset = val_dataset, test_dataset=test_dataset,
                                                                                               model=model,
                                                                                               device=device, T=T_train,
                                                                                               tau=tau, batch_size=batch_size,
                                                                                               loss=loss, num_nodes=num_nodes,
                                                                                               num_edges=num_edges,
                                                                                               scheduler=scheduler, pred_steps= pred_steps)


                    df_new = pd.DataFrame(data=[[str(model.__class__.__name__), tau, pred_steps, ev_list[0], ev_list[1], ev_list[2], ev_list[3], ev_list[4], ev_list[5]]],
                                          columns=['model', 'step', 'pred', 'mse', 'mae', 'mape', 'accuracy', 'r2',
                                                   'var'])
                    df_em = pd.concat([df_em, df_new], axis=0, ignore_index=True)


                    df = pd.DataFrame(columns= ['train_loss', 'val_loss','epoch_time'])
                    df['train_loss'] = train_loss_collections
                    df['val_loss'] = val_loss_collections
                    df['epoch_time'] = epoch_time
                    path = './train/'+str(model.__class__.__name__)+'/'+str(T)+'_'+str(T_train)+'/'+str(tau)+'_steps_for_pred'+str(pred_steps)+'_result/'
                    filename = 'exps_'+str(exp+1)+'.csv'
                    isExist = os.path.exists(path)
                    if not isExist:
                        os.makedirs(path)
                    df.to_csv(path + filename)
                    print('完成实验：' + path+filename)


                    path = './result/' + str(model.__class__.__name__)+'/'+str(T)+'_'+str(T_train)+'/'+str(tau)+'_steps_for_pred'+str(pred_steps)+'_result/'
                    isExist = os.path.exists(path)
                    filename = 'exps_'+str(exp+1)+'.csv'
                    if not isExist:
                        os.makedirs(path)
                    df_new.to_csv(path + filename)
                    print('完成：' + path + filename)

            path = './result/' + str(model.__class__.__name__) + '/' + str(T) + '_' + str(T_train) + '/' + str(
                tau) + '_steps/'  + '_result/'
            isExist = os.path.exists(path)
            filename = 'exps_' + str(exp + 1) + '.csv'
            if not isExist:
                os.makedirs(path)
            df_em.to_csv(path + filename)
            print('完成：' + path + filename)
            y_hat_name = 'day_' + str(int(T_train / 288) + 2) + '_pred.pt'
            torch.save(y_hat_test,'./y_hat/' + y_hat_name)

df_split = pd.read_csv('processed/split.csv')
Experiment([2],'./',[6],[1],1,df_split)