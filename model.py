import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Classifier(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout):
        super(Classifier,self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*1,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()



    def forward(self,list_mat,layer_norm,list_ind):

        list_out = list()
        #Select matrices
        for ind, ind_m in enumerate(list_ind):
            tmp_out = self.fc1[ind_m](list_mat[ind])
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)

            list_out.append(tmp_out)

        device = list_out[0].get_device()
        final_mat = torch.zeros_like(list_out[0]).cuda(device)
        for mat in list_out:
            final_mat += mat

        final_mat = final_mat/len(list_mat)

        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)


        return F.log_softmax(out, dim=1)


class Selector(nn.Module):
    def __init__(self,mask_size,nhidden):
        super (Selector,self).__init__()
        self.fc1 = nn.Linear(mask_size,nhidden)
        self.fc2 = nn.Linear(nhidden,1)
        self.act_fn = nn.ReLU()

    def forward(self,mask,loss):
        out = self.fc1(mask)
        out = self.act_fn(out)
        out = F.dropout(out,0.5,training=self.training)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    pass






