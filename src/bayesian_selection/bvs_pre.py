###################################################################
###### Latent Class Regression with Group Variable Selection ######
###################################################################

# Data Processing Tools
import pandas as pd
import numpy as np

# Progress Bar Visualization
from tqdm.auto import tqdm
import time

# Math Objects
from numpy import nan as NULL
from scipy.stats import invgamma, uniform, norm, dirichlet, rv_discrete

# Inherit base object from Interface
from .base import BaseSelection 

class BayesianVariableSelection(BaseSelection):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Constructor of BayesianVariableSelection
        
        Parameter
        =========
        X: word occurrance matrix
        y: dependent variable (e.g. avg. rating)
        """
        self.X = X
        self.y = y
    
    def _initialize_posterior(self):
        ## Instore posterior sampling
        ## Rules: Save x P x K
        Save, P, K = self.iteration, self.P, self.K
        N = self.N
        m = 1
        
        self.post_alp = NULL
        self.post_z = np.zeros(shape = (Save, P, K))
        self.post_mu = np.zeros(shape = (Save, P, K))
        self.post_tau = np.zeros(shape = (Save, P))
        self.post_w = list()
        self.post_sig_2 = list()
        self.post_d = np.zeros(shape = (Save, K))
        self.post_px = list()
        self.post_H = np.zeros(shape = (Save, N))
        self.post_nk = np.zeros(shape = (Save, K))
        self.post_p = np.zeros(shape = (m, N))
                
        ## Neighborhood savings
        self.setid_T = np.zeros(shape = (Save, P, P, K))

    def _initialize_hyperparameters(self, hyperparameters):
        # Configure Hyperparameters about Priors
        if hyperparameters == None:
            self.sp1, self.sp2, self.r1, self.r2 = (1, 1, 1, 0.01)
            self.J = 0.01
        else:
            self.sp1 = hyperparameters['sp1']
            self.sp2 = hyperparameters['sp2']
            self.r1 = hyperparameters['r1']
            self.r2 = hyperparameters['r2']
            self.J = hyperparameters['J']
    
    def _initialize_estimates(self):
        P, K = self.P, self.K
        theta = np.ones(K)
        d = theta / np.sum(theta)
        
        self.tau_2 = np.full(
            fill_value = invgamma(self.sp1, scale = self.sp2).rvs(1),
            shape = P
        )
        self.sig_2 = invgamma(self.r1, scale = self.r2).rvs(1)
        
        self.w = 0.5
        z = np.where(
            uniform().rvs(size = (P, K)) > self.w, 1, 0
        )
        mu = norm().rvs(size = (P, K))
        H = np.random.choice(
            range(K), size = self.N, replace = True, p = d
        )
        
        ordered = self.reorder(mu, H, z, d)
        self.pack_main_estimates(ordered['mu'], ordered['H'], ordered['z'], ordered['d'])
        
    
    def initialize(self, segment_no, iteration = 2000, burnin = 2000, cutoff = 0.3,
                  hyperparameters: list = None):
        """Initialize parameters for word variable selection
        
        Parameter
        ========
        segment_no: # of latent segment
        iteration: # of iteration during MCMC
        burnin: # of iteration to drop out
        cutoff: percentile to classify co-occurrance word groups
        hyperparameters: about setting priors
        """
        # Model configuration
        self.K = segment_no
        self.cutoff = cutoff
        self.N, self.P = self.X.shape
        
        # MCMC iteration information
        self._initialize_iteration(iteration, burnin)

        # Posterior savings
        self._initialize_posterior()
        
        # Configure hyperparamters
        self._initialize_hyperparameters(hyperparameters)

        # Initialize estimates
        self._initialize_estimates()
    
    def _unique(self, l, K):
        _, nk = np.unique(l, axis = 0, return_counts = True)
        if len(_) < K:
            nk = np.concatenate([nk, np.zeros(K - len(_))])
        
        return _, nk
    
    def _calculate_squ_term(self):
        X = self.X
        P, K = self.P, self.K
        mu, H, z, d = self.unpack_main_estimates()
        
        V = np.zeros(shape = (P, K))
        
        for k in range(K):
            idk = np.argwhere(H == k).reshape(-1)
            V[:, k] = (X[idk, :] ** 2).sum(axis = 0)        

        return V

    def unpack_main_estimates(self):
        return self.mu, self.H, self.z, self.d
    
    def pack_main_estimates(self, mu, H, z, d):
        self.mu = mu
        self.H = H
        self.z = z
        self.d = d
    
    def sample_sig2(self):
        mu, H, z, d = self.unpack_main_estimates()
        
        res = 0
        for k in range(self.K):
            idk = np.argwhere(H == k).reshape(-1)
            res += np.power(
                self.y[idk] - self.X[idk, :] @ mu[:, k], 2
            ).sum()
            
        sig_alpha = self.N / 2 + self.r1
        sig_beta = res / 2 + self.r2
        sig_2 = invgamma(sig_alpha, scale = sig_beta).rvs(1)
        
        return sig_2
    
    def sample_z_mu(self):
        # Recall Dimensions & Input Datas
        X, y = self.X, self.y
        V = self.V
        P = self.P
        sig_2, tau_2 = self.sig_2, self.tau_2
        mu, H, z, d = self.unpack_main_estimates()
        
        # Define nbhd_ids
        nbhd_ids = np.zeros(shape = (self.P, self.P, self.K))
        
        # Binding X
        X1 = np.clip(self.X, 0, 1)
        
        for k in range(self.K):
            idk = np.argwhere(H == k).reshape(-1)
            dist_mat = self.distance(X1[idk, :])
            
            for j in range(P):
                dis_mm = dist_mat[j, :]
                thq = np.quantile(dis_mm, self.cutoff)
                
                # index where data is seleceted by 'nbhd'
                set_id = np.where(dis_mm < thq)[0] 
                set_values = z[set_id, k] if len(set_id) > 0 else np.array([])
                Zk_1 = np.sum(set_values == 1)
                
                if len(set_values) > 0:
                    nbhd_ids[j, :len(set_id), k] = set_id
                
                eps = y[idk] - np.dot(X[idk, :][:, np.arange(P) != j], mu[np.arange(P) != j, k])
                tmp_mean = np.sum(eps * X[idk, j]) / sig_2
                tmp_var = V[j, k] / sig_2 + 1 / tau_2[j]
                
                logr = (-self.J * Zk_1) + \
                        0.5 * np.log(V[j, k] * tau_2[j] / sig_2 + 1) - \
                        (tmp_mean ** 2 / (2 * tmp_var))
                logr = np.clip(logr, a_min = -np.inf, a_max = 100)
                
                # Draw Z by log_r 
                z[j, k] = 1 if np.random.uniform() < 1 / (1 + np.exp(logr)) else 0

                # Sample mu
                if z[j, k] == 0:
                    mu[j, k] = 0
                else:
                    bjmu = tmp_mean / tmp_var
                    mu[j, k] = np.random.normal(bjmu, 1 / np.sqrt(tmp_var))
                    
        return z, mu, nbhd_ids
                
    def sample_tau_2(self, z, mu):
        a_w = z.sum(axis = 1)
        b_w = (mu ** 2).sum(axis = 1)
        
        a_t = a_w / 2 + self.sp1
        b_t = np.diag((mu ** 2) @ z.T) / 2 + self.sp2
        
        tau_2 = invgamma(a_t, scale = b_t).rvs(self.P)
        
        return tau_2
    
    def sample_d(self, nk):
        d = np.random.dirichlet(nk + np.ones(self.K)).reshape(-1)

        return d
    
    def sample_H(self, mu, d):
        # To broadcast calculation, use np.tile on Y and transpose it
        full_resid = np.tile(self.y, reps = (self.K, 1)).T - self.X @ mu
        wp = np.exp(-full_resid ** 2 / (2 * self.sig_2)) * d

        # sample H
        H = np.array([np.random.choice(self.K, p = prob / prob.sum()) for prob in wp])    
        return H
        
    def estimation(self):
        # Recall Dimensions & Input datas
        P, K = self.P, self.K
        X, y = self.X, self.y
        mu, H, z, d = self.unpack_main_estimates()
        
        # Calculate # of data point in each seg K 
        _, nk = self._unique(H, K)
        
        # Calculate variance
        self.V = self._calculate_squ_term()
        
        # Sample sigma_2
        self.sig_2 = self.sample_sig2()
        
        # Sample mu and z
        z, mu, nbhd_ids = self.sample_z_mu()
        
        # sample tau_2
        self.tau_2 = self.sample_tau_2(z, mu)
        
        # sample d
        d = self.sample_d(nk)
        
        # sample H
        H = self.sample_H(mu, d)
        
        # Reorder & save
        new_param = self.reorder(mu, H, z, d)
        self.pack_main_estimates(
            new_param['mu'], 
            new_param['H'], 
            new_param['z'], 
            new_param['d']
        )
        self.nk = nk
        self.nbhd_ids = nbhd_ids
    
    def saving(self, index):
        self.post_z[index, :, :] = self.z
        self.post_mu[index, :, :] = self.mu
        self.post_tau[index, :] = self.tau_2
        
        self.post_d[index, :] = self.d
        self.post_H[index, :] = self.H
        self.post_nk[index, :] = self.nk
        self.setid_T[index, :, :, :] = self.nbhd_ids        
        
        self.post_w.append(self.w)
        self.post_sig_2.append(self.sig_2)
        
    def finalize(self):
        self.posterior = {
            'z': np.array(self.post_z),
            'mu': np.array(self.post_mu),
            'tau': np.array(self.post_tau),
            'd': np.array(self.post_d),
            'H': np.array(self.post_H),
            'nk': np.array(self.post_nk),
            'nbhd': np.array(self.setid_T),
            'sig': np.array(self.post_sig_2)
        }
        
        
    def export_data(self) -> pd.DataFrame:
        """
        Export data used in the code, in forms of pandas dataframe
        """
        X = pd.DataFrame(self.X)
        y = pd.DataFrame(self.y)
        
        return (X, y)