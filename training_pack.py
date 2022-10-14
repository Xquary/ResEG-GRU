import time
import torch
from torch import nn as nn
import gcn_basic_pack as gbp
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import math
from torch_geometric.data import InMemoryDataset, Data, HeteroData
import numpy.linalg as la
import numpy as np
"""重构edge_index"""
def evaluation(a, b):
    a=a.detach().cpu()
    b =b.detach().cpu()
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

def train_for_Res(epochs,optimizer, dataloader, model, device, T, tau, batch_size, loss, num_nodes, num_edges, scheduler= None):#for graph
    loss_collections=[]
    start_time = time.time()
    for epoch in range(epochs):
        batch = 0
        for x, edge_index, edge_fixed, edge_attr in dataloader:
            time_pin1 = time.time()
            x = x.to(device)
            x = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
            edge_index = rc_edge_index(edge_index)
            edge_index = edge_index.long().to(device)
            edge_attr = edge_attr.to(device)
            edge_attr = edge_attr.reshape((edge_attr.shape[0]*edge_attr.shape[1],edge_attr.shape[2]))
            edge_fixed = edge_fixed.to(device)
            edge_fixed = edge_fixed.reshape((edge_fixed.shape[0]*edge_fixed.shape[1],edge_fixed.shape[2]))
            y = gbp.process_data_labels(x_0 = x, T = batch_size, tau = tau, num_nodes = num_nodes, num_edges = num_edges)#生成序列标签
            y_hat = model(x=x, edge_index = edge_index, edge_f = edge_fixed, edge_attr = edge_attr,T = batch_size, tau = tau,num_nodes = num_nodes, num_edges = num_edges, device =device)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            l=loss(y_hat,y)
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

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # UsingPyTorchIn-Builtscheduler
                scheduler.step()
            else:
                # Usingcustomdefinedscheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

        #acc=accuary(pred,data.y)
        #print(f'epoch{epoch + 1},'f'loss:{l.sum() / len(data.y):f}','acc:',acc)
        #loss_collections.append(l.sum() / y.numel())
        print(f'epoch{epoch + 1},'f'loss:{l.sum() / y.numel():f}')
    end_time = time.time()
    total_time = round(end_time - start_time,2)
    print('total time:',total_time,'s')

def train_for_Res_Cudaloader(epochs, optimizer, dataloader, val_dataset, test_dataset, model, device, T, tau, pred_steps, batch_size, loss, num_nodes, num_edges, scheduler= None):#for graph
    train_loss_collections=[]
    val_loss_collections = []
    epochs_time = []

    x_val, edge_index_val, edge_fixed_val, edge_attr_val = val_dataset[0].to(device), val_dataset[1].to(device), val_dataset[2].to(device), val_dataset[3].to(device)
    x_val = x_val.reshape((x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
    edge_index_val = rc_edge_index(edge_index_val, device=device)
    edge_index_val = edge_index_val.long()
    edge_attr_val = edge_attr_val.reshape((edge_attr_val.shape[0] * edge_attr_val.shape[1], edge_attr_val.shape[2]))
    edge_fixed_val = edge_fixed_val.reshape((edge_fixed_val.shape[0] * edge_fixed_val.shape[1], edge_fixed_val.shape[2]))
    y_val = gbp.process_data_labels(x_0 = x_val, T = x_val.shape[0], tau = tau, pred_steps = pred_steps, num_nodes = num_nodes, num_edges = num_edges, device=device)

    x_test, edge_index_test, edge_fixed_test, edge_attr_test = test_dataset[0].to(device), test_dataset[1].to(device), \
                                                           test_dataset[2].to(device), test_dataset[3].to(device)
    x_test = x_test.reshape((x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
    edge_index_test = rc_edge_index(edge_index_test, device=device)
    edge_index_test = edge_index_test.long()
    edge_attr_test = edge_attr_test.reshape((edge_attr_test.shape[0] * edge_attr_test.shape[1], edge_attr_test.shape[2]))
    edge_fixed_test = edge_fixed_test.reshape(
        (edge_fixed_test.shape[0] * edge_fixed_test.shape[1], edge_fixed_test.shape[2]))
    y_test = gbp.process_data_labels(x_0=x_test, T=x_test.shape[0], tau=tau, pred_steps=pred_steps, num_nodes=num_nodes,
                                    num_edges=num_edges, device=device)

    start_time = time.time()
    for epoch in range(epochs):
        batch = 0
        epoch_start = time.time()
        for x, edge_index, edge_fixed, edge_attr in dataloader:
            time_pin1 = time.time()
            x = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
            edge_index = rc_edge_index(edge_index,device=device)
            edge_index = edge_index.long()
            edge_attr = edge_attr.reshape((edge_attr.shape[0]*edge_attr.shape[1],edge_attr.shape[2]))
            edge_fixed = edge_fixed.reshape((edge_fixed.shape[0]*edge_fixed.shape[1],edge_fixed.shape[2]))
            y = gbp.process_data_labels(x_0 = x, T = batch_size, tau = tau, pred_steps = pred_steps, num_nodes = num_nodes, num_edges = num_edges, device=device)#生成序列标签
            y_hat = model(x=x, edge_index = edge_index, edge_f = edge_fixed, edge_attr = edge_attr,T = batch_size, tau = tau,num_nodes = num_nodes, num_edges = num_edges, device =device)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            l=loss(y_hat,y)
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
        epoch_end = time.time()
        epochs_time.append(round(epoch_end - epoch_start,2))
        y_hat_val = model(x=x_val, edge_index = edge_index_val, edge_f = edge_fixed_val, edge_attr = edge_attr_val,T = x_val.shape[0], tau = tau,num_nodes = num_nodes, num_edges = num_edges, device =device)
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
        print(f'epoch{epoch + 1},'f'train_loss:{l.sum() / y.numel():f}',f'val_loss:{val_loss.sum() / y_val.numel():f}')
        train_loss_collections.append(l.mean().cpu().detach().numpy())
        val_loss_collections.append(val_loss.mean().cpu().detach().numpy())

    #测试集上指标
    y_hat_test = model(x=x_test, edge_index=edge_index_test, edge_f=edge_fixed_test, edge_attr=edge_attr_test,
                      T=x_test.shape[0], tau=tau, num_nodes=num_nodes, num_edges=num_edges, device=device)
    mse, mae, mape, acc, r2, var = evaluation(y_test, y_hat_test)
    end_time = time.time()
    total_time = round(end_time - start_time,2)
    print('total time:',total_time,'s')
    return train_loss_collections, val_loss_collections, epochs_time, [mse, mae, mape, acc, r2, var], y_hat_test

def train_for_GRU_Cudaloader(epochs,optimizer, dataloader, test_dataset,  model, device, T, tau, batch_size, loss, num_nodes, num_edges, scheduler= None):#for graph
    train_loss_collections=[]
    val_loss_collections = []
    epochs_time = []
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
        epoch_start = time.time()
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
        epoch_end = time.time()
        epochs_time.append(round(epoch_end - epoch_start, 2))
        y_hat_val = model(embedding_val)
        val_loss = loss(y_hat_val, y_val)
        mse, mae, mape, acc, r2, var = evaluation(y_val, y_hat_val)
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
    return train_loss_collections, val_loss_collections,epochs_time,[mse, mae, mape, acc, r2, var],y_hat_val



