import torch
import torch.nn as nn
import torch.nn.functional as F
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, lower_init_lambda=5, upper_init_lambda=15, a_prior=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-1.2, 1.2))
        self.weight_rho = nn.Parameter(-9 + 0.1* torch.randn(out_features,in_features))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) 
        self.sigma_prior = (self.mu_prior+30).to(DEVICE)
        
        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(lower_init_lambda, upper_init_lambda))
        self.alpha = torch.empty(size=self.lambdal.shape)

        # model priors = Bernoulli(0.10)
        self.alpha_prior =  torch.zeros(out_features, in_features, device=DEVICE) + a_prior

     

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9+ .1 * torch.randn(out_features))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 30).to(DEVICE)

        # # bias model variational parameters
        # self.bias_lambdal = nn.Parameter(torch.Tensor(out_features).uniform_(lower_init_lambda, upper_init_lambda))
        # self.bias_alpha = torch.empty(size=self.bias_lambdal.shape)

        # # bias model priors = Bernoulli(0.10)
        # self.bias_alpha_prior =  torch.zeros(out_features, device=DEVICE) + a_prior

      
        # scalars
        self.kl = 0
     
      

    # forward path
    def forward(self, input, ensemble=True, sample=False, calculate_log_probs=False, post_train=False):
        '''
        NOTE: Adjusted forward function to fit Lars KMNIST implementation. 
        To get the "old" behaviour please see the "input_skip_connection" directory.
        '''
        self.alpha = 1 / (1 + torch.exp(-self.lambdal))
        if post_train:
            self.alpha = (self.alpha.detach() > 0.5) * 1.
            self.alpha_prior[self.alpha.detach() < 0.5] = 0. # Only include priors that is inlcuded in the model
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if ensemble or self.training:
            e_w = self.weight_mu * self.alpha
            var_w = self.alpha * (self.weight_sigma ** 2 + (1 - self.alpha) * self.weight_mu ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:  # median prob
            if sample:
                w = torch.normal(self.weight_mu, self.weight_sigma)
                b = torch.normal(self.bias_mu, self.bias_sigma)
            else:
                w = self.weight_mu
                b = self.bias_mu
            g = (self.alpha.detach() > 0.5) * 1.
            weight = w * g
            activations = torch.matmul(input, weight.T) + b
            if calculate_log_probs:
                self.alpha = g

        if self.training or calculate_log_probs:

            # # Want to avoid error when updating, hence I add a number close to zero
            # kl_bias = (self.bias_alpha * (torch.log((self.bias_sigma_prior / (self.bias_sigma+torch.tensor(1e-45)))+ torch.tensor(1e-45)) 
            #            - 0.5 + torch.log((self.bias_alpha / (self.bias_alpha_prior + torch.tensor(1e-45)))+torch.tensor(1e-45)) 
            #            + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) / (2 * self.bias_sigma_prior ** 2+torch.tensor(1e-45)))
            #            + (1-self.bias_alpha)*torch.log(((1-self.bias_alpha) / (1-self.bias_alpha_prior + torch.tensor(1e-45)))+torch.tensor(1e-45))).sum()

            # Want to avoid error when updating, hence I add a number close to zero
            kl_bias = (torch.log((self.bias_sigma_prior / (self.bias_sigma+torch.tensor(1e-45)))+ torch.tensor(1e-45)) 
                       - 0.5 + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) 
                       / (2 * self.bias_sigma_prior ** 2+torch.tensor(1e-45))).sum()
            
            # Do the same thing here
            kl_weight = (self.alpha * (torch.log((self.sigma_prior / (self.weight_sigma+torch.tensor(1e-45)))+torch.tensor(1e-45))
                                         - 0.5 + torch.log((self.alpha / (self.alpha_prior+torch.tensor(1e-45)))+torch.tensor(1e-45))
                                         + (self.weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2) / (
                                                 2 * self.sigma_prior ** 2+torch.tensor(1e-45)))
                         + (1 - self.alpha) * torch.log(((1 - self.alpha) / (1 - self.alpha_prior + torch.tensor(1e-45)))+torch.tensor(1e-45))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0

        return activations