# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
import os
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
import pandas as pd

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
        return '384_graphs_36_186.pt'

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

#%%
Data_raw = Getdata2('D:\py4fun\pyg\MyModel')

"""训练集"""
x=Data_raw.data.x
edge_attr = Data_raw.data.edge_attr
edge_fixed = Data_raw.data.edge_fixed
edge_index = Data_raw.data.edge_index


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


    labels = gbp.process_data_labels_3ds(x_0 = x[:num_training], T = num_training, tau= tau, pred_steps= pred_steps,num_nodes= num_nodes,num_edges= num_edges,device=device)
    x_new = gbp.process_data_x_GRU(x_raw= x[:num_training], T = num_training, tau = tau, num_nodes = num_nodes, pred_steps= pred_steps)
    edge_attr_new = gbp.process_data_x_GRU(x_raw= edge_attr[:num_training], T = num_training, tau = tau, num_nodes = num_edges, pred_steps= pred_steps)
    test_labels = gbp.process_data_labels_3ds(x[num_training:], T = T - num_training, tau =tau, pred_steps=pred_steps, num_nodes= num_nodes, num_edges= num_edges,device=device)
    test_x = gbp.process_data_x_GRU(x_raw= x[num_training:], T = T - num_training, tau = tau, num_nodes = num_nodes,pred_steps= pred_steps)
    test_edge_attr = gbp.process_data_x_GRU(x_raw= edge_attr[num_training:], T = T - num_training, tau = tau, num_nodes = num_edges, pred_steps= pred_steps)
    test_dataset = [test_x,test_edge_attr,test_labels]
    train_dataset = [x_new,edge_attr_new, labels]
    #dataloader = gbp.load_array((x_new, edge_attr_new, labels), batch_size= batch_size,is_train=False)
    return train_dataset, test_dataset
#%%

#%%
'''
def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY
'''

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


###### evaluation ######
def evaluation(a, b):
    mse_l = nn.MSELoss( reduce= 'mean')
    mse = mse_l(a,b)
    mae_l = nn.L1Loss()
    mae = mae_l(a,b)
    mape =np.fabs((a-b)/np.clip(a,0.1,1)).mean()
    a= a.numpy()
    b = b.numpy()
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return mse.numpy(), mae.numpy(), mape.numpy(), 1 - F_norm, r2, var


method = 'XGboost'  ####HA or SVR or ARIMA

def rc_x (x):
    """

    :param x_: [tau, num_nodes, num_channel]
    :return: x_new:[num_nodes,num_channels, tau]
    """
    x_new = torch.zeros((x.shape[1], x.shape[2], x.shape[0]))
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            x_new[i, j, :] = x[:,i,j]
    return x_new

########### HA #############
def Experiment(tau_list, pred_list, method_list,num_exp):
    df = pd.DataFrame(columns=['model','step','pred','mse','mae','mape','accuracy','r2','var'])
    for method in method_list:
        for tau in tau_list:

            for pred_steps in pred_list:

                for exp in range(num_exp):

                    days_step = 2
                    num_training = 11520 + days_step * 288 * exp
                    train_dataset, test_dataset = process_data_normalize_GRU(x=x, edge_attr=edge_attr,
                                                                             edge_fixed=edge_fixed, tau=tau, T=16992,
                                                                             num_nodes=69, num_edges=468,
                                                                             batch_size=288,
                                                                             num_training=num_training, pred_steps=pred_steps,
                                                                             device='cpu')
                    # %%
                    train_x = train_dataset[0]
                    train_labels = train_dataset[2]
                    test_x = test_dataset[0][:288]
                    test_labels = test_dataset[2][:288]
                    if method == 'MA':
                        y_hat = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        test_labels_new = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        for i in range(test_x.shape[0]):
                            x_ = rc_x(test_x[i])
                            test_labels_new[i] = rc_x(test_labels[i])
                            for num_n in range(x_.shape[0]):
                                for num_c in range(x_.shape[1]):
                                    for ps in range(pred_steps):
                                        if ps == 0:
                                            y_hat[i, num_n, num_c, ps] = x_[num_n, num_c, :].mean()
                                        else:
                                            y_hat[i, num_n, num_c, ps] = (x_[num_n, num_c, ps:].sum() + y_hat[i, num_n,
                                                                                                        num_c,
                                                                                                        :ps].sum()) / pred_steps
                        mse, mae, mape, acc, r2, var = evaluation(test_labels_new, y_hat)
                        print('HA_mse:%r' % mse,
                              'HA_mae:%r' % mae,
                              'HA_mape:%r' % mape,
                              'HA_acc:%r' % acc,
                              'HA_r2:%r' % r2,
                              'HA_var:%r' % var)


                    if method == 'HA':
                        alpha = 0.7
                        y_hat = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        test_labels_new = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        for i in range(test_x.shape[0]):
                            x_ = rc_x(test_x[i])
                            test_labels_new[i] = rc_x(test_labels[i])
                            for num_n in range(x_.shape[0]):
                                for num_c in range(x_.shape[1]):
                                    for ps in range(pred_steps):
                                        if ps == 0:
                                            y_hat[i, num_n, num_c, ps] = x_[num_n, num_c, -1] * alpha + x_[num_n, num_c, :-1].mean() * (1 - alpha)
                                        else:
                                            y_hat[i, num_n, num_c, ps] = (x_[num_n, num_c, ps:].sum() + y_hat[i, num_n,
                                                                                                        num_c,
                                                                                                        :ps - 1].sum()) / (pred_steps - 1) * (1 - alpha) + y_hat[i, num_n,
                                                                                                        num_c,
                                                                                                        ps - 1] *alpha
                        mse, mae, mape, acc, r2, var = evaluation(test_labels_new, y_hat)
                        print('HA_mse:%r' % mse,
                              'HA_mae:%r' % mae,
                              'HA_mape:%r' % mape,
                              'HA_acc:%r' % acc,
                              'HA_r2:%r' % r2,
                              'HA_var:%r' % var)
                    ############ SVR #############
                    if method == 'SVR':
                        train_x_new = torch.zeros(
                            (train_x.shape[0], train_x.shape[2], train_x.shape[3], train_x.shape[1]))
                        test_x_new = torch.zeros((test_x.shape[0], test_x.shape[2], test_x.shape[3], test_x.shape[1]))
                        test_labels_new = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        train_labels_new = torch.zeros((train_labels.shape[0], train_labels.shape[2],
                                                        train_labels.shape[3], train_labels.shape[1]))
                        y_hat = torch.zeros(
                            (test_labels.shape[0] * test_labels.shape[2] * test_labels.shape[3], test_labels.shape[1]))
                        for i in range(test_x.shape[0]):
                            test_x_new[i] = rc_x(test_x[i])
                            test_labels_new[i] = rc_x(test_labels[i])

                        for i in range(train_x.shape[0]):
                            train_x_new[i] = rc_x(train_x[i])
                            train_labels_new[i] = rc_x(train_labels[i])

                        y = torch.zeros((test_labels_new.shape[0] * test_labels_new.shape[1] * test_labels_new.shape[2],
                                         test_labels_new.shape[-1]))

                        for i in range(train_x_new.shape[1]):
                            t_x = train_x_new[:, i, :, :].reshape((train_x_new[:, i, :, :].shape[0] *
                                                                   train_x_new[:, i, :, :].shape[1],
                                                                   train_x_new[:, i, :, :].shape[2]))
                            t_y = train_labels_new[:, i, :, :].reshape((train_labels_new[:, i, :, :].shape[0] *
                                                                        train_labels_new[:, i, :, :].shape[1],
                                                                        train_labels_new[:, i, :, :].shape[2]))
                            v_x = test_x_new[:, i, :, :].reshape((test_x_new[:, i, :, :].shape[0] *
                                                                  test_x_new[:, i, :, :].shape[1],
                                                                  test_x_new[:, i, :, :].shape[2]))
                            v_y = torch.zeros((v_x.shape[0], pred_steps))
                            svr_model = SVR(kernel='linear')

                            for ps in range(pred_steps):
                                if ps == 0:
                                    t_y_p = t_y[:, ps]
                                    svr_model.fit(t_x, t_y_p)
                                    pre = svr_model.predict(v_x)
                                    v_y[:, ps] = torch.tensor(pre)
                                else:
                                    t_y_p = t_y[:, ps]
                                    t_x_p = torch.cat((t_x[:, ps:], t_y[:, :ps]), dim=1)
                                    svr_model.fit(t_x_p, t_y_p)
                                    v_x_p = torch.cat((v_x[:, ps:], v_y[:, :ps]), dim=1)
                                    pre = svr_model.predict(v_x)
                                    v_y[:, ps] = torch.tensor(pre)
                            y_hat[i * test_labels.shape[0] * test_labels.shape[-1]:(i + 1) * test_labels.shape[0] *
                                                                                   test_labels.shape[-1], :] = v_y
                            y[i * test_labels.shape[0] * test_labels.shape[-1]:(i + 1) * test_labels.shape[0] *
                                                                               test_labels.shape[-1],
                            :] = test_labels_new[:, i, :, :].reshape((test_labels_new[:, i, :, :].shape[0] *
                                                                      test_labels_new[:, i, :, :].shape[1],
                                                                      test_labels_new[:, i, :, :].shape[-1]))

                        mse, mae, mape, acc, r2, var = evaluation(y, y_hat)
                        print('SVR_rmse:%r' % mse,
                              'SVR_mae:%r' % mae,
                              'SVR_mape:%r' % mape,
                              'SVR_acc:%r' % acc,
                              'SVR_r2:%r' % r2,
                              'SVR_var:%r' % var)

                    ######## ARIMA #########
                    if method == 'ARIMA':
                        test_x = test_x.sum(3)
                        test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
                        test_x = test_x[:,:,10:16,:]
                        #test_x = test_x.sum(2)
                        #test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1, 1))

                        test_labels = test_labels.sum(3)
                        test_labels = test_labels.reshape((test_labels.shape[0], test_labels.shape[1], test_labels.shape[2], 1))
                        test_labels = test_labels[:, :, :, :]
                        #test_labels = test_labels.sum(2)
                        #test_labels = test_labels.reshape( (test_labels.shape[0], test_labels.shape[1], 1, 1))

                        y_hat = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2] * test_labels.shape[3], test_labels.shape[1]))
                        test_labels_new = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        for i in range(test_x.shape[0]):
                            x_ = rc_x(test_x[i])
                            x_ = x_.reshape((x_.shape[0] * x_.shape[1], x_.shape[-1]))
                            test_labels_new[i] = rc_x(test_labels[i])

                            rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
                            for j in range(x_.shape[0]):
                                model = ARIMA(x_[j].numpy(), order=[1, 0, 0])
                                properModel = model.fit()
                                predict_ts = properModel.predict(tau, tau + pred_steps - 1)
                                y_hat[i, j, :] = torch.tensor(predict_ts)
                        y_hat = y_hat.reshape((y_hat.shape[0] * y_hat.shape[1], y_hat.shape[-1]))
                        test_labels_new = test_labels_new.reshape((test_labels_new.shape[0] * test_labels_new.shape[1] *
                                                                   test_labels_new.shape[2], test_labels_new.shape[-1]))
                        mse, mae, mape, acc, r2, var = evaluation(test_labels_new, y_hat)
                        print('ARIMA_rmse:%r' % mse,
                              'ARIMA_mae:%r' % mae,
                              'ARIMA_mape:%r' % mape,
                              'ARIMA_acc:%r' % acc,
                              'ARIMA_r2:%r' % r2,
                              'ARIMA_var:%r' % var)
                        #    for i in range(109,num):
                        #        ts = data.iloc[:,i]
                        #        ts_log=np.log(ts)
                        #        ts_log=np.array(ts_log,dtype=np.float)
                        #        where_are_inf = np.isinf(ts_log)
                        #        ts_log[where_are_inf] = 0
                        #        ts_log = pd.Series(ts_log)
                        #        ts_log.index = a1
                        #        model = ARIMA(ts_log,order=[1,1,1])
                        #        properModel = model.fit(disp=-1, method='css')
                        #        predict_ts = properModel.predict(2, dynamic=True)
                        #        log_recover = np.exp(predict_ts)
                        #        ts = ts[log_recover.index]
                        #        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
                        #        rmse.append(er_rmse)
                        #        mae.append(er_mae)
                        #        acc.append(er_acc)
                        #        r2.append(r2_score)
                        #        var.append(var_score)

                    ######## KL #########
                    if method == 'XGboost':
                        train_x_new = torch.zeros(
                            (train_x.shape[0], train_x.shape[2], train_x.shape[3], train_x.shape[1]))
                        test_x_new = torch.zeros((test_x.shape[0], test_x.shape[2], test_x.shape[3], test_x.shape[1]))
                        test_labels_new = torch.zeros(
                            (test_labels.shape[0], test_labels.shape[2], test_labels.shape[3], test_labels.shape[1]))
                        train_labels_new = torch.zeros(
                            (
                            train_labels.shape[0], train_labels.shape[2], train_labels.shape[3], train_labels.shape[1]))
                        y_hat = torch.zeros(
                            (test_labels.shape[0] * test_labels.shape[2] * test_labels.shape[3], test_labels.shape[1]))
                        for i in range(test_x.shape[0]):
                            test_x_new[i] = rc_x(test_x[i])
                            test_labels_new[i] = rc_x(test_labels[i])

                        for i in range(train_x.shape[0]):
                            train_x_new[i] = rc_x(train_x[i])
                            train_labels_new[i] = rc_x(train_labels[i])

                        y = torch.zeros(
                            (test_labels_new.shape[0] * test_labels_new.shape[1] * test_labels_new.shape[2],
                             test_labels_new.shape[-1]))

                        for i in range(train_x_new.shape[1]):
                            t_x = train_x_new[:, i, :, :].reshape(
                                (train_x_new[:, i, :, :].shape[0] * train_x_new[:, i, :, :].shape[1],
                                 train_x_new[:, i, :, :].shape[2]))
                            t_y = train_labels_new[:, i, :, :].reshape((train_labels_new[:, i, :, :].shape[0] *
                                                                        train_labels_new[:, i, :, :].shape[1],
                                                                        train_labels_new[:, i, :, :].shape[2]))
                            v_x = test_x_new[:, i, :, :].reshape(
                                (test_x_new[:, i, :, :].shape[0] * test_x_new[:, i, :, :].shape[1],
                                 test_x_new[:, i, :, :].shape[2]))
                            v_y = torch.zeros((v_x.shape[0], pred_steps))
                            xgb_model = XGBRegressor(learning_rate=0.1)

                            for ps in range(pred_steps):
                                if ps == 0:
                                    t_y_p = t_y[:, ps]
                                    xgb_model.fit(t_x, t_y_p)
                                    pre = xgb_model.predict(v_x)
                                    v_y[:, ps] = torch.tensor(pre)
                                else:
                                    t_y_p = t_y[:, ps]
                                    t_x_p = torch.cat((t_x[:, ps:], t_y[:, :ps]), dim=1)
                                    xgb_model.fit(t_x_p, t_y_p)
                                    v_x_p = torch.cat((v_x[:, ps:], v_y[:, :ps]), dim=1)
                                    pre = xgb_model.predict(v_x)
                                    v_y[:, ps] = torch.tensor(pre)
                            y_hat[i * test_labels.shape[0] * test_labels.shape[-1]:(i + 1) * test_labels.shape[0] *
                                                                                   test_labels.shape[-1],
                            :] = v_y
                            y[i * test_labels.shape[0] * test_labels.shape[-1]:(i + 1) * test_labels.shape[0] *
                                                                               test_labels.shape[-1],
                            :] = test_labels_new[:, i, :, :].reshape((test_labels_new[:, i, :, :].shape[0] *
                                                                      test_labels_new[:, i, :, :].shape[1],
                                                                      test_labels_new[:, i, :, :].shape[-1]))

                        mse, mae, mape, acc, r2, var = evaluation(y, y_hat)
                        print('XGboost_mse:%r' % mse,
                              'XGboost_mae:%r' % mae,
                              'XGboost_mae:%r' % mape,
                              'XGboost_acc:%r' % acc,
                              'XGboost_r2:%r' % r2,
                              'XGboost_var:%r' % var)

                    df_new = pd.DataFrame(data=[[method, tau, pred_steps, mse, mae, mape, acc, r2, var]],
                                          columns=['model', 'step', 'pred', 'mse', 'mae', 'mape', 'accuracy', 'r2',
                                                   'var'])
                    df = pd.concat([df, df_new], axis=0, ignore_index=True)

    #path = './result/' + 'traditional/384_288/'
    path = './result/' + 'traditional/16992_11520/'
    isExist = os.path.exists(path)
    filename = str(tau)+'_steps.csv'
    if not isExist:
        os.makedirs(path)
    df.to_csv(path + filename)
    print('完成：' + path + filename)

#Experiment(tau_list=[6], pred_list=[1,2,3,4,5,6], method_list=["HA","SVR","XGboost"])
Experiment(tau_list=[6], pred_list=[6], method_list=["ARIMA"], num_exp = 1)