import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

TEMPER_PRIOR = 0.001
# NOTE: mps is not yet available when using the current implementation.
#       Could be that it will be availble in later versions of PyTorch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gaussian:
    '''
    NOTE: This will be the main class for the Gaussian distribution, however, we 
          will aslo create more classes that uses this class as "super". This
          is done to make the code more readable.
    '''
    def __init__(self, mu, rho) -> None:
        self.mu = mu
        self.rho = rho
        # Reparametrization trick; now mu and rho are not random, but 
        # instead trainable parameters.
        self.normal = torch.distributions.Normal(0,1) # Used to sample from
    
    @property
    def sigma(self):
        '''
        NOTE: $\sigma = \log(1 + \exp(\rho))$. Makes it easier to 
              optimize sigma as \rho can take any value, while 
              keeping \sigma positiv
        '''
        return torch.log1p(torch.exp(self.rho)+1e-8)
    
    def rsample(self):
        '''
        Here, self.sigma is obtained from the "getter" property of 
        function sigma.
        '''
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob_iid(self, input):
        return (-math.log(math.sqrt(2*math.pi)) - torch.log(self.sigma + 1e-8)
                - ((input-self.mu)**2) / (2 * self.sigma**2 + 1e-8))
    
    def log_prob(self,input):
        return self.log_prob_iid(input).sum()
    

class GaussianSpikeSlab(Gaussian):
    '''
    The spike-and-slab prior for the weights, where we 
    use a normal distribution if gamma = 1, and a value
    close to zero if gamma = 0

    NOTE: Added this class to seperate spike-and-slab and 
          the Gaussian dist.
    '''
    def __init__(self, mu, rho) -> None:
        super().__init__(mu, rho)

    def full_log_prob(self, input, gamma):
        return (torch.log(gamma * (torch.exp(self.log_prob_iid(input)))
                          + (1 - gamma) + 1e-8)).sum() # We add 1e-8 to no be exactly equal to zero
    
class Bernoulli:
    '''
    Used as prior for \gamma (the indicator variable)
    '''
    def __init__(self, alpha) -> None:
        self.alpha = alpha
        self.exact = False

    def rsample(self):
        if self.exact:
            gamma = torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)
        else:
            # Concrete distirbution
            gamma = torch.distributions.RelaxedBernoulli(probs=self.alpha, temperature=TEMPER_PRIOR).rsample()
        
        return gamma

    def log_prob(self, gamma):
        if self.exact:
            gamma = torch.round(gamma.detach())
        return (gamma*torch.log(self.alpha+1e-8) + 
                (1-gamma)*torch.log(1-self.alpha+1e-8)).sum()   

class GaussGamma:
    '''
    Prior for \mu and \sigma (or \rho, which we use in the Gaussian class).
    This is a "hyperprior" for the prior distribution of the weights.

    NOTE: Will (mainly) use Gauss-Gamma distribution when \gamma equal 1. When
          \gamma equal 0, we will set the hyperprior to zero (I think, now he sets it
          to be a number higher than 1, but I have implemented with 0).
    '''
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
        self.exact = False
        # Initialize gamma distribution. Can note that 1/sigma ~ Gamma(a,b) in
        # this case. 
        self.one_over_sigma = torch.distributions.Gamma(self.a, self.b)
    
    def log_prob(self, input, gamma):
        '''
        input here is the weights.
        Gamma is included in this part as we will only use
        GaussGamma distirbution if we include the weight. We would 
        set the prior to zero (or no distirbution) if we do not want
        to include 

        NOTE: I have changed this part. 
        TODO: Ask Aliaksandr if it seems correct.
        '''
        # tau = 1/sigma
        tau = self.one_over_sigma.rsample()
        if self.exact:
            gamma = torch.round(gamma.detach())

        # return (
        #     gamma*(self.a*torch.log(self.b) + (self.a - 0.5)*tau - self.b*tau
        #             - tau*torch.pow(input,2)*0.5 - torch.lgamma(self.a) 
        #             - 0.5*torch.log(torch.tensor(2*np.pi))) +
        #     (1-gamma)*torch.log(torch.tensor(1e-8)) # unsure if we need this part, could we not just set to zero?
        # ).sum()

        return (
            gamma*(self.a*torch.log(self.b) + (self.a - 0.5)*tau - self.b*tau
                    - tau*torch.pow(input,2)*0.5 - torch.lgamma(self.a) 
                    - 0.5*torch.log(torch.tensor(2*np.pi))) +
            (1-gamma)*1e-8 # unsure if we need this part, could we not just set to zero?
        ).sum()
    
# class Beta:
#     '''
#     Prior for \gamma. 
#     This is also a "hyperprior", but now for the prior distribution of the 
#     indicator variable \gamma.

#     NOTE: Will try to use the Beta distribution, as this is whats decribed in the
#           paper, but if it do not work, I will have to chang eit back to the 
#           BetaBinomial distribution. 
#     '''
#     def __init__(self, pa, pb) -> None:
#         self.pa = pa
#         self.pb = pb
#         self.exact = False

#     def log_prob(self, gamma, pa, pb):
#         '''
#         Input here should be alpha; the probability of including a variable.
#         I think this will be reflected in gamma.
#         '''
#         if self.exact:
#             gamma = torch.round(gamma.detach())
#         return (
#             (self.pa - 1)*torch.log(gamma) + (self.pb - 1)*torch.log(1-gamma) -
#             (torch.lgamma(self.pa) + torch.lgamma(self.pb) - torch.lgamma(self.pa + self.pb))
#         ).sum()

# define BetaBinomial distibution
class BetaBinomial(object):
    '''
    Used as a prior for gamma. I have not checked the math for this one, so could
    that this one is perfectly fine, and maybe even more optimized than 
    the original "plan". However, I think I should make my LBBNN such that we 
    seperate between using Beta, and seperate thatwe use Bernoulli. Think this would make 
    it easier to work with later when making a more costume network (skip connections 
    and stuff like that). How this "new" implementation will look like is for later
    worries.
    TODO: Seperate into a version more "true" to the paper implementation (if feasible)
    NOTE: Could be that it is okay. See Beta-Binomial wiki, think it expisitly says that
          we can do it in this fashion.
    TODO: Ask Aliksandr, or Lars, why the BetaBinomial distribution where used here, and not
          the Beta distribution.  
    '''
    def __init__(self, pa, pb):
        super().__init__()
        self.pa = pa
        self.pb = pb
        self.exact = False

    def log_prob(self, input, pa, pb):
        if self.exact:
            gamma = torch.round(input.detach())
        else:
            gamma = input
        return (torch.lgamma(torch.ones_like(input)) + torch.lgamma(gamma + torch.ones_like(input) * self.pa)
                + torch.lgamma(torch.ones_like(input) * (1 + self.pb) - gamma) + torch.lgamma(
                    torch.ones_like(input) * (self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa + gamma)
                - torch.lgamma(torch.ones_like(input) * 2 - gamma) - torch.lgamma(
                    torch.ones_like(input) * (1 + self.pa + self.pb))
                - torch.lgamma(torch.ones_like(input) * self.pa) - torch.lgamma(torch.ones_like(input) * self.pb) + 1e-8).sum()

    def rsample(self):
        gamma = torch.distributions.RelaxedBernoulli(
            probs=torch.distributions.Beta(self.pa, self.pb).rsample().to(DEVICE), temperature=0.001).rsample().to(
            DEVICE)
        return gamma
    

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        
        # Configuration of the layer 
        self.in_features = in_features
        self.out_features = out_features

        # Weight prior paramters initialization. 
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01,0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features, in_features))
        self.weight = GaussianSpikeSlab(self.weight_mu, self.weight_rho)


        # Weight hyperpriors initialization
        self.weight_a = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        self.weight_b = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        self.weight_prior = GaussGamma(self.weight_a, self.weight_b)


        # model prior parameters
        # NOTE: We will use \alpha = 1/(1+\exp(-\lambda)). This forces \alpha between 0 and 1 and 
        #       makes the model more flexible when training as \lambda can take any value when being updated
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-10,10))
        self.alpha = torch.Tensor(out_features, in_features).uniform_(0.999,0.9999)
        self.gamma = Bernoulli(self.alpha)

        # model hyperpriors 
        self.pa = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        self.pb = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        self.gamma_prior = BetaBinomial(pa=self.pa, pb=self.pb)

        # bias (intercept) for parameter priors
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1*torch.randn(out_features))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # # Test to see if this makes any difference
        # self.bias_mu = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        # self.bias_rho = nn.Parameter(torch.Tensor(1).uniform_(1,1.1))
        # self.bias = GaussGamma(self.bias_mu, self.bias_rho)


        # bias (intercept) for hyperpriors
        self.bias_a = nn.Parameter(torch.Tensor(out_features).uniform_(1,1.1))
        self.bias_b = nn.Parameter(torch.Tensor(out_features).uniform_(1,1.1))
        self.bias_prior = GaussGamma(self.bias_a, self.bias_b)


        # Scalars used for calcualting loss
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, cgamma, sample=False, calculate_log_probs=False):
        # for sampling
        if self.training or sample:
            ws = self.weight.rsample()
            weight = cgamma*ws
            # weight = ws # Try to not include gamma
            bias = self.bias.rsample()


        else:
            weight = self.alpha*self.weight.mu
            # weight = self.weight.mu  # Try to not include alpha
            bias = self.bias.mu

        # Calcualte the KL 
        if self.training or calculate_log_probs:
            self.alpha = 1 / (1 + torch.exp(-self.lambdal))
            self.log_prior = self.weight_prior.log_prob(weight, cgamma) \
                            + self.bias_prior.log_prob(bias, torch.ones_like(bias)) \
                            + self.gamma_prior.log_prob(cgamma, pa=self.pa, pb=self.pb)
            
            self.log_variational_posterior = self.weight.full_log_prob(input=weight, gamma=cgamma) \
                            + self.gamma.log_prob(cgamma) \
                            + self.bias.log_prob(bias)
            
        else:
            self.log_prior, self.log_variational_posterior = 0,0
        
        return F.linear(input, weight, bias)