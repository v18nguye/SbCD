import math
import pdb
import numpy as np
import torch
    
    
class ConstP1Rate():
    def __init__(self, data_cfg, device):
        '''constant forward rate inspired by
        https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py
        '''
        self.device = device
        self.S = 231 # 230 + 1 dummy dim 
        self.rate_const = data_cfg.rate_const
        base_rate =  np.zeros((self.S, self.S))
        base_rate[:,1] =1
        base_rate = base_rate - np.diag(np.ones(self.S)) # Q_b matrix
        eigvals, eigvecs = np.linalg.eig(base_rate)
        inv_eigvecs = np.linalg.inv(eigvecs)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)
        self.base_rate = torch.from_numpy(base_rate).float().to(self.device)

    def _rate_scalar(self, t):
        '''get p_t(x) is constant for all t
        '''
        return self.rate_const

    def _integral_rate_scalar(self, t):
        '''get \int_0^t p_s(x)ds
        '''
        return self.rate_const*t
    
    def rate(self, t):
        '''get Q_t = p_t(x)*Q_b
        Note: we can replace Q_b (corresponding r(x_t|x_0)) by Q_t (by masking self-transition) for sampling x_t at t.
        They are equivalent up to the scalar p_t(x).
        '''
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)
        return torch.tile(rate_scalars*self.base_rate.view(1,S,S), (B, 1, 1))
    
    def sampling_holding_time(self, t):
        '''posterior holding time
        '''
        rand_u = torch.rand_like(t)
        rate = self._rate_scalar(t)
        rate_int = self._integral_rate_scalar(t)
        return -torch.log(1-rand_u*(1-torch.exp(-rate_int)))/rate
    
    def transition(self, t):
        '''get P_{t|0}(x|x_0) = D@\exp(\Lamda*\int_0^t p_s(x)ds)@D^{-1}
        where Q_b = D@\Lamda@D^{-1}
        '''
        B, S = t.shape[0], self.S
        transitions = self.eigvecs.view(1, S, S) @\
                        torch.diag_embed(torch.exp(self.eigvals.view(1,S)*self._integral_rate_scalar(t).view(B,1))) @\
                        self.inv_eigvecs.view(1, S, S)
        
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] ConstP1Rate, large negative transition values {torch.min(transitions)}")

         # effective moving to P1, otherwise moves to other space groups even with very small probs
        transitions[transitions < 1e-8] = 0.0
        return transitions

     
def get_forward_rate(rate_function_name, data_cfg, device):
    
    if rate_function_name == 'const':
        return ConstP1Rate(data_cfg, device)
    else:
        raise ValueError(rate_function_name)
