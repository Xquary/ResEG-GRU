import torch
from torch import nn as nn
import gcn_basic_pack as gbp
import torch_geometric as pyg
from torch_geometric import nn as pyg_nn
import torch.functional as F
"""ResEG_GRU"""
class ResEG(nn.Module):
    def __init__(self ,net_ResEG ,**kwargs):
        super(ResEG, self).__init__(**kwargs)
        self.net_ResEG = net_ResEG

    def forward(self, x, edge_index, edge_f, edge_attr ,device):
        edge_index = edge_index.long()
        data= self.net_ResEG((x, edge_index, edge_f, edge_attr ,device))
        x_1, e_1 = data[0], data[3]
        return x_1,e_1

class Encoder(nn.Module):
    def __init__(self, num_nodes, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.en_gru = nn.GRU(input_size= 5*num_nodes, hidden_size=256, num_layers=2, dropout= 0.5 ,batch_first=True)

    def forward(self, embedding):
        h,state= self.en_gru(embedding)
        # y_hat,_ = self.de_gru(y_hat)
        return h


class Decoder(nn.Module):
    def __init__(self, tau, pred_steps, num_nodes, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.de_gru = nn.GRU(input_size=256, hidden_size=64, dropout=0.5, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(in_features=64 * tau, out_features=64 * tau),nn.ReLU(),
                                 nn.Linear(in_features=64 * tau, out_features=pred_steps*num_nodes*5)
                                 )
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
    def forward(self, h):
        x_hat, state = self.de_gru(h)
        x_hat = x_hat.reshape((x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2]))
        x_hat = self.mlp(x_hat)
        x_hat = x_hat.reshape((x_hat.shape[0], self.pred_steps, self.num_nodes, 5))
        # y_hat,_ = self.de_gru(y_hat)
        return x_hat


class EncoderDecoder(nn.Module):
    def __init__(self, tau, pred_steps, num_nodes,**kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = Encoder(num_nodes)
        self.decoder = Decoder(tau= tau, pred_steps=pred_steps,num_nodes = num_nodes)

    def forward(self, embedding):
        y_hat = self.encoder(embedding)
        y_hat = self.decoder(y_hat)
        return y_hat


class ResEG_DE_Model(nn.Module):
    def __init__(self, net_ResEG, tau, pred_steps, num_nodes, **kwargs):
        super(ResEG_DE_Model, self).__init__(**kwargs)
        self.res = ResEG(net_ResEG)
        self.ED = EncoderDecoder(tau= tau, pred_steps = pred_steps, num_nodes = num_nodes)
        self.pred_steps = pred_steps

    def forward(self, x, edge_index, edge_f, edge_attr, T, tau, num_nodes, num_edges, device):
        x, edge_attr = self.res(x=x, edge_index=edge_index, edge_f=edge_f, edge_attr=edge_attr, device=device)
        x, edge_attr = gbp.process_data_x(x_raw=x, edge_attr_raw=edge_attr, T=T, tau=tau, num_nodes=num_nodes,
                                          num_edges=num_edges, device=device, pred_steps= self.pred_steps)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        edge_attr = edge_attr.reshape((edge_attr.shape[0], edge_attr.shape[1], edge_attr.shape[2] * edge_attr.shape[3]))
        y_hat = self.ED(x)
        return y_hat


"""Basic Two_layer GRU"""

class Encoder_GRU(nn.Module):
    def __init__(self,num_nodes, **kwargs):
        super(Encoder_GRU, self).__init__(**kwargs)
        self.en_gru = nn.GRU(input_size= 5*num_nodes, hidden_size= 256, dropout = 0.5, num_layers= 2,batch_first=True)
        #self.mlp = nn.Linear(in_features= 180, out_features= 20)
    def forward(self, embedding):

        h,state= self.en_gru(embedding)
        #h = self.mlp(embedding)
        # y_hat,_ = self.de_gru(y_hat)
        return h


class Decoder_GRU(nn.Module):
    def __init__(self,tau, pred_steps, num_nodes, **kwargs):
        super(Decoder_GRU, self).__init__(**kwargs)
        self.de_gru = nn.GRU(input_size=256, hidden_size=16, dropout= 0.5, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(in_features=16 * tau, out_features=num_nodes*5*pred_steps))
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
    def forward(self, h):
        x_hat, state = self.de_gru(h)
        #x_hat=h
        x_hat = x_hat.reshape((x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2]))
        x_hat = self.mlp(x_hat)
        x_hat = x_hat.reshape((x_hat.shape[0],self.pred_steps, self.num_nodes, 5))
        # y_hat,_ = self.de_gru(y_hat)
        return x_hat


class EncoderDecoder_GRU(nn.Module):
    def __init__(self, tau, pred_steps, num_nodes, **kwargs):
        super(EncoderDecoder_GRU,self).__init__(**kwargs)
        self.encoder = Encoder_GRU(num_nodes= num_nodes)
        self.decoder = Decoder_GRU(tau = tau,pred_steps=pred_steps, num_nodes=num_nodes)

    def forward(self, embedding):
        y_hat = self.encoder(embedding)
        y_hat = self.decoder(y_hat)
        return y_hat

"""Basic Two_layer LSTM"""

class Encoder_LSTM(nn.Module):
    def __init__(self, num_nodes,**kwargs):
        super(Encoder_LSTM, self).__init__(**kwargs)
        self.en_gru = nn.LSTM(input_size= 5*num_nodes, hidden_size= 128, dropout = 0.5, num_layers= 2,batch_first=True)
        #self.mlp = nn.Linear(in_features= 180, out_features= 20)
    def forward(self, embedding):

        h,state= self.en_gru(embedding)
        #h = self.mlp(embedding)
        # y_hat,_ = self.de_gru(y_hat)
        return h


class Decoder_LSTM(nn.Module):
    def __init__(self, tau, pred_steps,num_nodes,**kwargs):
        super(Decoder_LSTM, self).__init__(**kwargs)
        self.de_gru = nn.LSTM(input_size=128, hidden_size=16, dropout= 0.5, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(in_features=16 * tau, out_features=pred_steps*num_nodes*5))
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
    def forward(self, h):
        x_hat, state = self.de_gru(h)
        #x_hat=h
        x_hat = x_hat.reshape((x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2]))
        x_hat = self.mlp(x_hat)
        x_hat = x_hat.reshape((x_hat.shape[0], self.pred_steps, self.num_nodes, 5))
        # y_hat,_ = self.de_gru(y_hat)
        return x_hat


class EncoderDecoder_LSTM(nn.Module):
    def __init__(self, tau,pred_steps,num_nodes,**kwargs):
        super(EncoderDecoder_LSTM, self).__init__(**kwargs)
        self.encoder = Encoder_LSTM(num_nodes)
        self.decoder = Decoder_LSTM(tau=tau, pred_steps= pred_steps, num_nodes=num_nodes)

    def forward(self, embedding):
        y_hat = self.encoder(embedding)
        y_hat = self.decoder(y_hat)
        return y_hat

"""GCN_GRU"""
class GCN_blk(nn.Module):
    def __init__(self,**kwargs):
        super(GCN_blk, self).__init__(**kwargs)
        self.net_GCN = pyg_nn.Sequential('x, edge_index', [(pyg_nn.GCNConv(in_channels = 5, out_channels = 8), 'x, edge_index -> x'), (pyg_nn.GCNConv(in_channels = 8, out_channels = 5), 'x, edge_index -> x') ])
        #self.net_GCN = pyg_nn.GCN(in_channels = 5, hidden_channels = 8 ,out_channels = 5, num_layers = 2)
    def forward(self, x, edge_index, edge_f, edge_attr):
        edge_index = edge_index.long()
        x= self.net_GCN(x, edge_index)
        return x

class Encoder_GCN_GRU(nn.Module):
    def __init__(self, num_nodes,  **kwargs):
        super(Encoder_GCN_GRU, self).__init__(**kwargs)
        self.en_gru = nn.GRU(input_size= 5*num_nodes, hidden_size=256, dropout= 0.5 ,batch_first=True)

    def forward(self, embedding):
        h,state= self.en_gru(embedding)
        # y_hat,_ = self.de_gru(y_hat)
        return h


class Decoder_GCN_GRU(nn.Module):
    def __init__(self,tau, pred_steps, num_nodes,**kwargs):
        super(Decoder_GCN_GRU, self).__init__(**kwargs)
        self.de_gru = nn.GRU(input_size=256, hidden_size=128, dropout=0.5, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(in_features=128 * tau, out_features=64*tau), nn.ReLU(),
                                 nn.Linear(in_features=64*tau, out_features=num_nodes*5*pred_steps))
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
    def forward(self, h):
        x_hat, state = self.de_gru(h)
        x_hat = x_hat.reshape((x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2]))
        x_hat = self.mlp(x_hat)
        x_hat = x_hat.reshape((x_hat.shape[0], self.pred_steps, self.num_nodes, 5))
        # y_hat,_ = self.de_gru(y_hat)
        return x_hat


class EncoderDecoder_GCN_GRU(nn.Module):
    def __init__(self, tau, pred_steps,num_nodes,**kwargs):
        super(EncoderDecoder_GCN_GRU, self).__init__(**kwargs)
        self.encoder = Encoder_GCN_GRU(num_nodes=num_nodes)
        self.decoder = Decoder_GCN_GRU(tau=tau, pred_steps=pred_steps, num_nodes=num_nodes)

    def forward(self, embedding):
        y_hat = self.encoder(embedding)
        y_hat = self.decoder(y_hat)
        return y_hat


class GCN_DE_Model(nn.Module):
    def __init__(self,  tau, pred_steps, num_nodes,**kwargs):
        super(GCN_DE_Model, self).__init__(**kwargs)
        self.GCN = GCN_blk()
        self.ED = EncoderDecoder_GCN_GRU(tau=tau, pred_steps= pred_steps,num_nodes = num_nodes)
        self.pred_steps = pred_steps

    def forward(self, x, edge_index, edge_f, edge_attr, T, tau, num_nodes, num_edges, device):
        x = self.GCN(x=x, edge_index=edge_index, edge_f=edge_f, edge_attr=edge_attr)
        x, edge_attr = gbp.process_data_x(x_raw=x, edge_attr_raw=edge_attr, T=T, tau=tau, num_nodes=num_nodes,
                                          num_edges=num_edges, device=device, pred_steps= self.pred_steps)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        y_hat = self.ED(x)
        return y_hat


"""Edgeconv_GRU"""
class Edgeconv_blk(nn.Module):
    def __init__(self,**kwargs):
        super(Edgeconv_blk, self).__init__(**kwargs)
        self.mlp = nn.Linear(2*5, 5)
        self.net_Edgeconv = pyg_nn.EdgeConv(nn =self.mlp)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index, edge_f, edge_attr):
        edge_index = edge_index.long()
        x= self.net_Edgeconv(x, edge_index)
        return x

class Encoder_Edgeconv_GRU(nn.Module):
    def __init__(self,num_nodes, **kwargs):
        super(Encoder_Edgeconv_GRU, self).__init__(**kwargs)
        self.en_gru = nn.GRU(input_size= 5*num_nodes, hidden_size=256, dropout= 0.5 ,batch_first=True)

    def forward(self, embedding):
        h,state= self.en_gru(embedding)
        # y_hat,_ = self.de_gru(y_hat)
        return h


class DecoderEdgeconv_GRU(nn.Module):
    def __init__(self,tau, pred_steps, num_nodes,**kwargs):
        super(DecoderEdgeconv_GRU, self).__init__(**kwargs)
        self.de_gru = nn.GRU(input_size=256, hidden_size=4, dropout=0.5, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(in_features=4 * tau, out_features=16*tau), nn.ReLU(),
                                 nn.Linear(in_features=16*tau, out_features= num_nodes*pred_steps*5))
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
    def forward(self, h):
        x_hat, state = self.de_gru(h)
        x_hat = x_hat.reshape((x_hat.shape[0], x_hat.shape[1] * x_hat.shape[2]))
        x_hat = self.mlp(x_hat)
        x_hat = x_hat.reshape((x_hat.shape[0], self.pred_steps, self.num_nodes, 5))
        # y_hat,_ = self.de_gru(y_hat)
        return x_hat


class EncoderDecoderEdgeconv_GRU(nn.Module):
    def __init__(self,tau, pred_steps, num_nodes, **kwargs):
        super(EncoderDecoderEdgeconv_GRU, self).__init__(**kwargs)
        self.encoder = Encoder_Edgeconv_GRU(num_nodes=num_nodes)
        self.decoder = DecoderEdgeconv_GRU(tau=tau,pred_steps= pred_steps,num_nodes=num_nodes)

    def forward(self, embedding):
        y_hat = self.encoder(embedding)
        y_hat = self.decoder(y_hat)
        return y_hat


class Edgeconv_DE_Model(nn.Module):
    def __init__(self, tau, pred_steps, num_nodes, **kwargs):
        super(Edgeconv_DE_Model, self).__init__(**kwargs)
        self.EG = Edgeconv_blk()
        self.ED = EncoderDecoderEdgeconv_GRU(tau=tau,pred_steps=pred_steps,num_nodes=num_nodes)
        self.pred_steps = pred_steps
    def forward(self, x, edge_index, edge_f, edge_attr, T, tau, num_nodes, num_edges, device):
        x = self.EG(x=x, edge_index=edge_index, edge_f=edge_f, edge_attr=edge_attr)
        x, edge_attr = gbp.process_data_x(x_raw=x, edge_attr_raw=edge_attr, T=T, tau=tau, num_nodes=num_nodes, pred_steps= self.pred_steps,
                                          num_edges=num_edges, device=device)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        edge_attr = edge_attr.reshape((edge_attr.shape[0], edge_attr.shape[1], edge_attr.shape[2]*edge_attr.shape[3]))
        #x = torch.cat((x,edge_attr),dim =2)
        y_hat = self.ED(x)
        return y_hat