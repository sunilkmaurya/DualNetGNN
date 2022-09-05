from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
import pickle
import copy
import itertools
from collections import defaultdict
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.02, help='Learning rate Scalar')
parser.add_argument('--lr_sel',type=float, default=0.01, help='Learning rate for selector')
parser.add_argument('--wd_sel',type=float,default=1e-05,help='weight decay selector layer')
parser.add_argument('--step1_iter',type=int, default=400, help='Step-1 iterations')
parser.add_argument('--step2_iter',type=int, default=20, help='Step-2 iterations')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#total feature matrices to be generated
num_layer = 2*args.layer+1

#maximum length of subset to find
feat_select = 4
sec_iter = args.step2_iter

layer_norm = bool(int(args.layer_norm))
print("==========================")
print(f"Dataset: {args.data}")
#print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
#print(f" w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, w_sel:{args.wd_sel}, lr_fc:{args.lr_fc}, lr_sel:{args.lr_sel}, 1st step iter: {args.step1_iter}, 2nd step iter: {args.step2_iter}")

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'



def train_step(model,optimizer,labels,list_mat,idx_train,list_ind):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm,list_ind)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,list_mat,idx_val,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()


def test_step(model,labels,list_mat,idx_test,list_ind):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #print(mask_val)
        return loss_test.item(),acc_test.item()



def selector_step(model,optimizer_sel,mask,o_loss):
    model.train()
    optimizer_sel.zero_grad()
    mask.requires_grad = True
    output = model(mask,o_loss)
    selector_loss = 10*F.mse_loss(output,o_loss)
    selector_loss.backward()
    input_grad = mask.grad.data
    optimizer_sel.step()
    return selector_loss.item(), input_grad


def selector_eval(model,mask,o_loss):
    model.eval()
    with torch.no_grad():
        output = model(mask,o_loss)
        selector_loss = F.mse_loss(output,o_loss)
        return selector_loss.item()


def new_optimal_mask(model, model_sel, optimizer_sel, idx_val, list_mat, device,labels):

    #Calculate input gradients
    equal_masks = torch.ones(num_layer).float().to(device)
    #Assign same weight to all indices
    equal_masks *= 0.5
    model_sel.train()
    optimizer_sel.zero_grad()
    equal_masks.requires_grad = True
    output = model_sel(equal_masks,None)
    output.backward()
    tmp_grad = equal_masks.grad.data
    tmp_grad = torch.abs(tmp_grad)

    #Top mask indices by gradients
    best_grad = sorted(torch.argsort(tmp_grad)[-feat_select:].tolist())

    #Creating possible optimal subsets with top mask indices
    new_combinations = list()
    for ll in range(1,feat_select+1):
        new_combinations.extend(list(itertools.combinations(best_grad,ll)))

    list_ind = list(range(len(new_combinations)))
    
    best_mask = []
    best_mask_loss = []
    #From these possible subsets, sample and check validation loss
    for _ in range(10):
        get_ind = random.choices(list_ind)[0]
        get_ind = list(new_combinations[get_ind])
        get_ind = sorted(get_ind)
        best_mask.append(get_ind)
        input_mat = [list_mat[ww] for ww in get_ind]

        loss_val,acc_val = validate_step(model,labels,input_mat,idx_val,get_ind)
        best_mask_loss.append(loss_val)


    #Find indices with minimum validation loss
    min_loss_ind = np.argmin(best_mask_loss)
    optimal_mask = best_mask[min_loss_ind]


    return optimal_mask, model_sel, model



def train(datastr,splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    features = features.to(device)

    adj = adj.to(device)
    adj_i = adj_i.to(device)
    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    #Calculate all aggregated features using A and A+I
    for ii in range(args.layer):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)


    model = Classifier(nfeat=num_features,
                #nlayers=2*args.layer + 1,
                nlayers=num_layer,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout).to(device)


    optimizer_sett_classifier = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
    ]

    optimizer = optim.Adam(optimizer_sett_classifier)

    model_sel = Selector(num_layer,args.hidden).to(device)
    optimizer_select = [
        {'params':model_sel.fc1.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel},
        {'params':model_sel.fc2.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel}
    ]
    optimizer_sel = optim.Adam(optimizer_select)

    bad_counter = 0
    best = 999999999
    best_sub = []


    #Calculate all possible combinations of subsets upto length feat_select
    combinations = list()
    for nn in range(1,feat_select+1):
        combinations.extend(list(itertools.combinations(range(num_layer),nn)))


    dict_comb = dict()
    for kk,cc in enumerate(combinations):
        dict_comb[cc] = kk
    
    #Step-1 training: Exploration step 

    for epoch in range(args.step1_iter):
        #choose one subset randomly
        rand_ind = random.choice(combinations)
        #create input to model
        input_mat = [list_mat[ww] for ww in rand_ind]

        #Train classifier and selector
        loss_tra,acc_tra = train_step(model,optimizer,labels,input_mat,idx_train,rand_ind)
        loss_val,acc_val = validate_step(model,labels,input_mat,idx_val,rand_ind)

        #Input mask vector to selector
        input_mask = torch.zeros(num_layer).float().to(device)
        input_mask[list(rand_ind)] = 1.0
        input_loss = torch.FloatTensor([loss_tra]).to(device)
        eval_loss = torch.FloatTensor([loss_val]).to(device)
        loss_select, input_grad = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
        #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)


    #Starting Step-2: Exploitation
    dict_check_loss = dict()
    for epoch in range(args.epochs):

        if epoch<sec_iter:
            #Upto sec_iter epoches optimal subsets are identified
            train_mask, model_sel, model = new_optimal_mask(model, model_sel, optimizer_sel, idx_val, list_mat,device, labels)
            

        if epoch==sec_iter:
            min_ind = min(list(dict_check_loss.keys()))
            train_mask = dict_check_loss[min_ind]


        input_mat = [list_mat[ww] for ww in train_mask]
        loss_tra,acc_tra = train_step(model,optimizer,labels,input_mat,idx_train,train_mask)
        loss_val,acc_val = validate_step(model,labels,input_mat,idx_val,train_mask)


        dict_check_loss[loss_val] = train_mask

        if epoch < sec_iter:
            input_mask = torch.zeros(num_layer).float().to(device)
            input_mask[list(train_mask)] = 1.0
            input_loss = torch.FloatTensor([loss_tra]).to(device)
            eval_loss = torch.FloatTensor([loss_val]).to(device)
            loss_select, _ = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
            #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)

        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))

        '''


        if loss_val < best and epoch>= sec_iter:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
            best_sub = train_mask

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break


    select_ind = best_sub


    input_mat = [list_mat[ww] for ww in select_ind]

    test_out = test_step(model,labels,input_mat,idx_test,select_ind)
    acc = test_out[1]


    return acc*100

t_total = time.time()
acc_list = []


for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accuracy_data = train(datastr,splitstr)
    acc_list.append(accuracy_data)


print("Train time: {:.4f}s".format(time.time() - t_total))
print(f"Test accuracy: {np.mean(acc_list):.2f}, {np.round(np.std(acc_list),2)}")

import os
os.remove(checkpt_file)
