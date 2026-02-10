###################################################################
###### Latent Class Regression with Group Variable Selection ######
###################################################################

# Data Processing Tools
import pandas as pd
import numpy as np

# Progress Visualization
import ipywidgets as widgets
from IPython.display import display 
from tqdm.auto import tqdm

# Output Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Math Objects
import time
from numpy import nan as NULL
from scipy.stats import invgamma, uniform, norm, dirichlet, rv_discrete
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist, pdist, squareform

from statsmodels.api import OLS

# Utils
from .utils import summary_to_df, find_cluster_members

# Inherit base object from Interface
from .base import BaseSelection 

class BayesianVariableSelection(BaseSelection):
    def __init__(self, X: pd.DataFrame, y: pd.Series, weight = None, dist_type = 'cooccur', track = False):
        """Constructor of BayesianVariableSelection
        
        Parameter
        =========
        X: word occurrance matrix
        y: dependent variable (e.g. avg. rating)
        """
        self.X = X
        self.y = y
        self.dist_type = dist_type
        self.track = track
        
        if isinstance(weight, type(None)):
            weight = np.ones(y.shape)
        
        self.weight = weight
        
        if dist_type == 'correlate':
            self.X = self.X / weight[:, np.newaxis]
            # Intercept = 1
            self.X[:, 0] = 1

    def diagnose_model(self, sig = 0.05):
        """Diagnose whether the correlation between X and y are fertile to apply the correlation-based variable selection model.
        The method exploits statsmodels.regression.linear_model.OLS
        
        Parameter
        =========
        sig: significance score
        """
        result = OLS(exog = self.X, endog = self.y).fit().summary()
        
        model_param = pd.DataFrame(result.tables[1].data).iloc[1:, :]
        model_param.columns = ['variable', 'parameter', 'std_err', 't_stat', 'p_value', 'CI_low', 'CI_upp']
        model_param['p_value'] = model_param['p_value'].apply(lambda x: float(x))

        model_param['variable'] = np.linspace(0, len(model_param) - 1, len(model_param))
        model_param['variable'] = model_param['variable'].astype(int)

        # Remove Intercept
        return model_param.loc[(model_param['p_value'] < sig) & (model_param['variable'] > 0), :]
        
    def _on_slider_change(self, change):
        idx = change["new"]
        with self.track_fig.batch_update():
            posterior_z = np.array(self.post_z)[:idx].mean(axis = 0)
            for k in range(self.K):
                self.track_fig.data[k].x = posterior_z[:, k]
            
    def _initialize_visual(self):
        # Variable Selection Tracker
        if self.track == True:
            initial_z = np.zeros(shape = (self.P, self.K))
            self.track_fig = go.FigureWidget(
                px.histogram(initial_z, 
                             template = 'simple_white', 
                             width = 800, height = 500, nbins = 50, 
                             range_y = (0, self.P), range_x = (0, 1), 
                             barmode = 'group')
            )
            self.slider = widgets.IntSlider(value=0, min=0, max=0, step=1,
                           description='Iter', continuous_update=True)
            self.slider.observe(self._on_slider_change, names = 'value')
            
            self.track_fig.update_layout(title = "Real-time Variable Selection Distribution Tracker")
            
            out = widgets.Output()
            with out:                                 # out 안에서만 display
                display(self.track_fig)
            
            display(widgets.VBox([out, self.slider]))
            
    def _initialize_posterior(self):
        ## Instore posterior sampling
        ## Rules: Save x P x K
        Save, P, K = self.iteration, self.P, self.K
        N = self.N
        m = 1
        
        self.post_alp = list()
        self.post_z = list()
        self.post_mu = list()
        self.post_tau = list()
        self.post_w = list()
        self.post_sig_2 = list()
        self.post_d = list()
        self.post_px = list()
        self.post_H = list()
        self.post_nk = list()
        self.post_p = list()
                
        ## Neighborhood savings
        self.setid_T = list()
        
        self.p_dim = None
        self.joint_D = None

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
                
    
    def initialize(self, segment_no, iteration = 2000, burnin = 2000, cutoff = 0.3, step = 10, random_seed = None,
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
        self._initialize_iteration(iteration, burnin, step, random_seed)

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
        nbhd_ids = np.zeros(shape = (self.P, self.P, self.K)) - 1
        
        X1 = np.clip(self.X, 0, 1)
        
        for k in range(self.K):
            idk = np.argwhere(H == k).reshape(-1)
            
            if self.dist_type == 'cooccur':
                # self.distance is deprecated. Use scipy.spatial.distance
                # dist_mat = self.distance(X1[idk, :])
                
                dist_mat = squareform(pdist(X1[idk, :].T))
            elif self.dist_type == 'correlate':
                dist_mat = np.corrcoef(X[idk, :], rowvar = False)
                dist_mat[np.isnan(dist_mat)] = 0
                np.fill_diagonal(dist_mat, 1)
            else:
                raise ValueError(f"Non-supportive dist_type, name: {self.dist_type}")
            
            for j in range(P):
                dis_mm = dist_mat[j, :]
                thq = np.quantile(dis_mm, self.cutoff)
                
                # index where data is seleceted by 'nbhd'
                if self.dist_type == 'cooccur':
                    set_id = np.where(dis_mm < thq)[0] 
                elif self.dist_type == 'correlate':
                    set_id = np.where(dis_mm > thq)[0] 
                
                set_values = z[set_id, k] if len(set_id) > 0 else np.array([])
                Zk_1 = np.sum(set_values == 1)
                
                if len(set_id) > 0:
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
                    draw = np.random.normal(bjmu, 1 / np.sqrt(tmp_var))
                    
                    mu[j, k] = draw[0]
                    
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
        
        # What Error?
        full_resid = full_resid.astype(float)
        
        wp = np.exp(-full_resid ** 2 / (2 * self.sig_2)) * d
        
        # Error Handling...
        EPS = 1e-5
        wp = np.where(np.isnan(wp), EPS, wp)
        prob_total = wp.sum(axis = 1)
        wp[prob_total == 0] = np.ones(self.K) / self.K
        
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
        self.post_z.append(self.z)
        self.post_mu.append(self.mu)
        self.post_tau.append(self.tau_2)
        
        self.post_d.append(self.d)
        self.post_H.append(self.H)
        self.post_nk.append(self.nk)
        self.setid_T.append(self.nbhd_ids)
        
        self.post_w.append(self.w)
        self.post_sig_2.append(self.sig_2)
        
        # Reset Network Structure
        self.p_dim = None
        self.joint_D = None
        
        # Update Variable Selection Tracker
        if self.track == True:
            posterior_z = np.array(self.post_z).mean(axis = 0)
            with self.track_fig.batch_update():
                for k in range(self.K):
                    self.track_fig.data[k].x = posterior_z[:, k]
                self.slider.max += 1
                self.slider.value = self.slider.max
        
        
    def finalize(self):
        self.posterior = {
            'z': np.array(self.post_z),
            'mu': np.array(self.post_mu),
            'tau2': np.array(self.post_tau),
            'd': np.array(self.post_d),
            'H': np.array(self.post_H),
            'nk': np.array(self.post_nk),
            'nbhd': np.array(self.setid_T),
            'sig2': np.array(self.post_sig_2)
        }
        
        
        
    def drop_draws(self, burnin):
        step = self.step
        drop_index = burnin // step
        
        for key in self.posterior:
            value = self.posterior[key]
            value = value[drop_index:]
            self.posterior[key] = value
        
        return self.posterior
        
        
    def find_membership(self):
        membership = list()
        for ind in range(self.N):
            tmp = self.posterior['H'][:, ind]
            _, nk = np.unique(tmp, return_counts = True)
            membership.append(nk.argmax())
            
        return np.array(membership)
    
    def find_network_structure(self):
        if isinstance(self.joint_D, type(None)):
            joint_D = np.zeros(shape = (self.P, self.P, self.K))
            nbhd = self.posterior['nbhd']
            S, P, _, K = nbhd.shape

            # To Do: Parallel Computing!
            for l in range(P):
                mask = (nbhd == l)
                joint_D[:, l, :] = mask.sum(axis = (0, 2))

            for k in range(K):
                np.fill_diagonal(joint_D[:, :, k], self.iteration)
            self.joint_D = joint_D
        
        return self.joint_D
    
    def find_cooccurance_structure(self, cut = 0.95):
        S, P, _, K = self.posterior['nbhd'].shape

        joint_D = self.find_network_structure()
        p_dim = np.zeros(shape = (P, P, K)) - 1

        tt = self.posterior_mean('z')

        for k in range(K):
            sel_t = np.where(tt[:, k] > cut)[0]

            for j in range(P):
                pdim = np.where(joint_D[j, :, k] > cut * S)[0]
                pdim0 = np.intersect1d(pdim, sel_t, assume_unique=True)

                if len(pdim0) > 0:
                    p_dim[j, 0:len(pdim0), k] = list(pdim0)

        self.p_dim = p_dim
            
        return self.p_dim
        
    def find_topic_by_seg(self, k, p_dim, cut = 0.95):
        """Find topic by segment.
        Segment no: 0 ~ K-1"""
        symm_M = np.zeros(shape = (self.P, self.P))
        tt = self.posterior_mean('z')
        
        for j in range(self.P):
            tmp_i = p_dim[j, :, k]
            tmp_i = tmp_i[tmp_i != -1]
            symm_M[j, tmp_i.astype(int)] = 1
        
        # Remove intercept information
        sel_t = np.where(tt[:, k] > cut)[0]
        sel_t -= 1
        sel_t = sel_t[sel_t >= 0]
        self.select_ind = sel_t
        
        # Remove intercept information
        symm_M = symm_M[1:, :][:, 1:] 
        symm_M0 = symm_M[sel_t.astype(int).tolist(), :][:, sel_t.astype(int).tolist()]
        return symm_M0, sel_t
    
    def find_correlation_by_seg(self, seg_no, cut = 0.95):
        p_dim = self.find_cooccurance_structure(cut)
        symm_M0, label = self.find_topic_by_seg(seg_no, p_dim, cut)
        sel_t = self.select_ind
        
        select_mu = np.mean(
            np.array(self.post_mu),
            where = (np.array(self.post_z) == 1),
            axis = 0
        )[1:, seg_no]
        post_z = np.array(self.post_z).mean(axis = 0)[1:, seg_no]
        
        return pd.DataFrame(
            data = [
                post_z,
                select_mu,
                self.posterior_mean('mu')[1:, seg_no]
            ],
            index = ['select_ratio', 'coef|z=1', 'coef']
        ).transpose().reset_index()

    def find_group_members(self, seg_no, k = 2, cut = 0.95):
        p_dim = self.find_cooccurance_structure(cut)
        symm_M0, label = self.find_topic_by_seg(seg_no, p_dim, cut)
        
        Z = hierarchy.linkage(symm_M0, method = 'complete', metric = 'euclidean')
        
        cluster_candidate = Z[-(k - 1):, [0, 1]]
        cluster_no = len(Z) * 2
        cluster_index = list()
        cluster_dict = dict()
        
        for elem in Z[-(k-1):, [0, 1]].reshape(-1):
            if elem <= len(Z) * 2 -(k-1):
                cluster_index.append(elem)
        
        for index in cluster_index:
            cluster_dict[index] = find_cluster_members(Z, index)
        
        return cluster_dict
    
    def visualize_topic_seg(self, seg_no, k = 2, cut = 0.95):
        p_dim = self.find_cooccurance_structure(cut)
        symm_M0, label = self.find_topic_by_seg(seg_no, p_dim, cut)
        
        Z = hierarchy.linkage(symm_M0, method = 'complete', metric = 'euclidean')
        thres = Z[-(k-1), 2] - 1e-9

        fig = ff.create_dendrogram(X = symm_M0, labels = label, color_threshold = thres, linkagefun = lambda _: Z)\
                .update_layout(width = 600, height = 400, template = 'simple_white',
                        title = f"Dendrogram about Seg {seg_no}")
        
        return fig
    
    def export_data(self) -> pd.DataFrame:
        """
        Export data used in the code, in forms of pandas dataframe
        """
        X = pd.DataFrame(self.X)
        y = pd.Series(self.y)
        
        return (X, y)
    
    def resurrect_display(self):
        display(widgets.VBox([self.track_fig, self.slider]))
        
    def visualize_heatmap(self, seg_no, k = 2, cut = 0.95):
        p_dim = self.find_cooccurance_structure(cut)
        symm_M0, label = self.find_topic_by_seg(seg_no, p_dim, cut)
        
        Z = hierarchy.linkage(symm_M0, method = 'complete', metric = 'euclidean')
        thres = Z[-(k-1), 2] - 1e-9

        # Initialize figure by creating upper dendrogram
        fig = ff.create_dendrogram(symm_M0, orientation='bottom', labels = label, color_threshold = thres, linkagefun = lambda _: Z).update_layout(template = 'simple_white')
        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'

        # Create Side Dendrogram
        dendro_side = ff.create_dendrogram(symm_M0, orientation='right', labels = label, color_threshold = thres, linkagefun = lambda _: Z).update_layout(template = 'simple_white')
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'
        
        # Add Side Dendrogram Data to Figure
        for data in dendro_side['data']:
            fig.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        data_dist = pdist(symm_M0)
        heat_data = pd.DataFrame(squareform(data_dist), index = label, columns = label)

        heat_data = heat_data.loc[dendro_leaves.tolist(),:]
        heat_data = heat_data.loc[:,dendro_leaves.tolist()]
        
        heatmap = [
            go.Heatmap(
                x = dendro_leaves,
                y = dendro_leaves,
                z = heat_data,
                colorscale = 'inferno',
                colorbar = {
                    'x': 1.2
                }
            )
        ]
        
        heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
        
        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)
            
        # Edit Layout
        fig.update_layout({'width':800, 'height':800,
                                 'showlegend':False, 'hovermode': 'closest',
                                 },
            yaxis=dict(automargin=True),
            xaxis=dict(automargin=True))

        # # Edit xaxis
        fig.update_layout(xaxis={'domain': [.15, 1],
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'ticks':""})

        # Edit xaxis2
        fig.update_layout(xaxis2={'domain': [0, .135],
                                           'mirror': False,
                                           'showgrid': False,
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""})

        # Edit yaxis
        fig.update_layout(yaxis={'domain': [0, .85],
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'side': 'right',
                                          'showticklabels': True,
                                          'tickmode': "array",
                                          'tickvals': dendro_side['layout']['yaxis']['tickvals'],
                                          'ticktext': dendro_leaves
                                })

        # Edit yaxis2
        fig.update_layout(yaxis2={'domain':[.825, .975],
                                           'mirror': False,
                                           'showgrid': False,
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""})

        # Plot!
        return fig