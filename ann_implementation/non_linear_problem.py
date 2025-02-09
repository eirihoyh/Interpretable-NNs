import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from config_non_linear import config


import os
import sys

current_dir = os.getcwd()

sys.path.append('islbbnn')
import plot_functions as pf
import pipeline_functions as pip_func
sys.path.append('islbbnn/networks')
from ann_net import NeuralNet

HIDDEN_LAYERS = config['n_layers'] - 2 
epochs = config['num_epochs']
dim = config['hidden_dim']
n_nets = config['n_nets']
n_samples = config['n_samples']
lr = config['lr']
class_problem = config["class_problem"]
non_lin = config["non_lin"]
verbose = config['verbose']
save_res = config['save_res']
patience = config['patience']
reg_level = config['reg_level']


# Define BATCH sizes
BATCH_SIZE = int((n_samples*0.8)/5)
TEST_BATCH_SIZE = int(n_samples*0.10) # Would normally call this the "validation" part (will be used during training)
VAL_BATCH_SIZE = int(n_samples*0.10) # and this the "test" part (will be used after training)

TRAIN_SIZE = int((n_samples*0.80))
TEST_SIZE = int(n_samples*0.10) # Would normally call this the "validation" part (will be used during training)
VAL_SIZE = int(n_samples*0.10) # and this the "test" part (will be used after training)

NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

print(NUM_BATCHES)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


# Get linear data
y, X = pip_func.create_data_unif(n_samples, beta=[100,1,1,1,1], dep_level=0.0, classification=class_problem, non_lin=non_lin)

n, p = X.shape  # need this to get p 
print(n,p,dim)


# Split keep some of the data for validation after training
X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y)

test_dat = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)

# select the device and initiate model

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
all_nets = {}
metrics_several_runs = []
metrics_median_several_runs = []


for ni in range(n_nets):
    post_train = False
    print('network', ni)
    # Initate network
    torch.manual_seed(ni+42)
    net = NeuralNet(dim, p, HIDDEN_LAYERS, classification=class_problem, n_classes=1, act_func=F.relu).to(DEVICE)
    nr_weights = 0
    for name, param in net.named_parameters():
        for i in range(HIDDEN_LAYERS+1):
            if f"linears.{i}.weight" in name:
                nr_weights += np.prod(param.data.shape)
    print(nr_weights)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    
    scheduler = MultiStepLR(optimizer, milestones=[int(0.3*epochs), int(0.5*epochs), int(0.7*epochs), int(0.9*epochs)], gamma=0.2)

    all_nll = []
    all_loss = []

    # Split into training and test set
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=1/9, random_state=ni, stratify=y)
            
    train_dat = torch.tensor(np.column_stack((X_train,y_train)),dtype = torch.float32)
    val_dat = torch.tensor(np.column_stack((X_val,y_val)),dtype = torch.float32)

    # train_dat = torch.tensor(np.column_stack((X_train_original,y_train_original)),dtype = torch.float32)
    
    # Train network
    counter = 0
    highest_acc = 0
    best_model = copy.deepcopy(net)
    for epoch in range(epochs):
        if verbose:
            print(epoch)
        nll, loss = pip_func.train(net, train_dat, optimizer, BATCH_SIZE, NUM_BATCHES, p, DEVICE, nr_weights, post_train=post_train, multiclass=False, ann=True, reg_level=reg_level)
        pred_score, used_weights   = pip_func.val_ann(net, val_dat, DEVICE, verbose=verbose, reg=(not class_problem), multiclass=False)
        if pred_score >= highest_acc:
            counter = 0
            highest_acc = pred_score
            best_model = copy.deepcopy(net)
        else:
            counter += 1
        
        all_nll.append(nll)
        all_loss.append(loss)

        if counter >= patience:
            break

        scheduler.step()
    
    if save_res:
        torch.save(net, f"ann_implementation/network/net{ni}_non_linear")
    all_nets[ni] = net 
    # Results
    metrics, metrics_median = pip_func.test_ensemble_ann(all_nets[ni], test_dat, DEVICE, SAMPLES=100, CLASSES=1, reg=(not class_problem)) # Test same data 100 times to get average 
    metrics_several_runs.append(metrics)
    metrics_median_several_runs.append(metrics_median)
    pf.run_path_graph_weight_ann(net, save_path=f"ann_implementation/path_graphs/net{ni}_non_linear_relu", show=False)

if verbose:
    print(metrics)
m = np.array(metrics_several_runs)
m_median = np.array(metrics_median_several_runs)

print("Results full model:")
print(m)
print("\n\nResults median prob model:")
print(m_median)

if save_res:
    # m = np.array(metrics_several_runs)
    np.savetxt(f'ann_implementation/results/ann_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_non_lin_func_relu_full.txt',m,delimiter = ',')
    # m_median = np.array(metrics_median_several_runs)
    np.savetxt(f'ann_implementation/results/ann_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_non_lin_func_relu_median.txt',m_median,delimiter = ',')




