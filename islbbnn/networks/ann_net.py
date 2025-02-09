import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.lrt_layers import BayesianLinear

class NeuralNet(nn.Module):
    def __init__(
            self, 
            dim, 
            p, 
            hidden_layers, 
            classification=True, 
            n_classes=1, 
            act_func=torch.sigmoid, 
            thresh=0.005):
        super(NeuralNet, self).__init__()
        self.thresh = thresh
        self.p = p
        self.act_func = act_func
        self.multiclass = n_classes > 1
        self.classification = classification
        self.linears = nn.ModuleList([nn.Linear(p, dim, bias=False)])
        self.linears.extend([nn.Linear((dim + p), (dim), bias=False) for _ in range(hidden_layers-1)])
        self.linears.append(nn.Linear((dim+p), n_classes, bias=False))
        if classification:
            if not self.multiclass: 
                self.loss = nn.BCELoss(reduction='sum') # Setup loss (Binary cross entropy as binary classification)
            else:
                self.loss = nn.NLLLoss(reduction='sum')
        else:
            self.loss = nn.MSELoss(reduction='sum')     # Setup loss (Mean Squared Error loss as regression problem)

    def forward(self, x):
        x_input = x.view(-1, self.p)
        x = self.act_func(self.linears[0](x_input))
        i = 1
        for l in self.linears[1:-1]:
            x = self.act_func(l(torch.cat((x, x_input),1)))
            i += 1

        if self.classification:
            if self.multiclass:
                out = F.log_softmax((self.linears[i](torch.cat((x, x_input),1))), dim=1)
            else:
                out = torch.sigmoid(self.linears[i](torch.cat((x, x_input),1)))
        else:
            out = self.linears[i](torch.cat((x, x_input),1))
        return out
   
    def mpm(self, x):
        x_input = x.view(-1, self.p)
        w = self.linears[0].weight.clone().detach()
        w[abs(w)<self.thresh] = 0
        x = self.act_func(torch.matmul(x_input,w.T))
        i = 1
        for l in self.linears[1:-1]:
            w_l = l.weight.clone().detach()
            w_l[abs(w_l)<self.thresh] = 0
            x_ = torch.cat((x, x_input),1)
            preact = torch.matmul(x_,w_l.T)
            x = self.act_func(preact)
            i += 1
            
        
        w_out = self.linears[i].weight.clone().detach()
        w_out[abs(w_out) < self.thresh] = 0
        x_out = torch.cat((x, x_input),1)
        preact = torch.matmul(x_out,w_out.T)
        if self.classification:
            if self.multiclass:
                out = F.log_softmax((preact), dim=1)
            else:
                out = torch.sigmoid(preact)
        else:
            out = preact
        return out
    
    def forward_preact(self, x,sample=False,ensemble=False,calculate_log_probs=False,post_train=False):
        '''
        NOTE: Sample, ensemble, calculate_log_probs and post_train not used. These are 
        included such re-usage of code in pip_func would be easier (related to local explain). 
        '''
        x_input = x.view(-1, self.p)
        x = self.act_func(self.linears[0](x_input))
        i = 1
        for l in self.linears[1:-1]:
            x = self.act_func(l(torch.cat((x, x_input),1)))
            i += 1

        out = self.linears[i](torch.cat((x, x_input),1))
        return out
    
    def nr_of_used_weights(self):
        counter = 0
        for l in self.linears:
            w = l.weight.clone().detach()
            w[abs(w)<self.thresh] = 0
            # print(torch.sum(torch.abs(w)>0))
            counter += torch.sum(torch.abs(w)>0)

        return counter