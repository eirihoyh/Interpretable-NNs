#!/usr/bin/env python3
# -*- coding: utf-8 -*-
config = {}
config['num_epochs' ] = 5000  
config['lr'] = .0075
config['patience'] = 10000
config['n_nets'] = 10  # Number of different networks to run for one problem
config['n_layers'] = 4+2 # Two hidden + (one input+one output)
config['hidden_dim'] = 20 # Number of hidden nodes per layer
config['class_problem'] = True  # If classification problem or not
# For the linear and non-linear problem
config['n_samples'] = 4*10**4
config['non_lin'] = False  # Wanting to create a non-linear or linear dataset 

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = True  # If we should save the results