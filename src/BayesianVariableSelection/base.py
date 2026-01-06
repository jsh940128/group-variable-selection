import numpy as np
from tqdm.auto import tqdm

class BaseSelection(object):
    def __init__(self):
        pass
    
    def _initialize_iteration(self, iteration, burnin, step, random_seed = None):
        self.iteration = iteration
        self.burnin = burnin
        self.step = step
        
        if random_seed == None:
            pass
        else:
            np.random.seed(random_seed)

    def estimation(self):
        """Interface about estimation function
        
        MCMC iteration calls estimation() function when fit() function is called.
        You should override estimation function to work with the Selection object.
        """
        pass
        
    def saving(self, position):
        """Interface about saving function
        
        When the iteration exceed burnin phase, then posterior draws will be saved by calling saving() function.
        You should override saving function to save posterior results.
        """
        pass
    
    def finalize(self):
        """Interface about finalizing MCMC chains."""
        pass
    
    def _initialize_visual(self):
        """Interface about visualizing MCMC chains"""
        pass
    
    def fit(self):
        """Conduct MCMC iteration.
        
        Inherited object need to edit estimation() and saving() function.
        """
        self._initialize_visual()
        
        # MCMC Iteration Procedure
        for index in tqdm(range(self.iteration + self.burnin)):
            self.estimation()
            
            if (index >= self.burnin) & ((index - self.burnin) % self.step == 0):
                self.saving((index - self.burnin) / self.step)
        
        self.finalize()
        
    def update(self, iteration):
        """Update and save the posterior result with new iteration without initialize.
        Note: If the dimension for each saving objects are already specified, then the update function won't work.
        """
        for index in tqdm(range(iteration)):
            self.iteration += 1
            self.estimation()
            
            if index % self.step == 0:
                self.saving(index)
            
        self.finalize()
    
    def reorder(self, mu, H, z, d):
        """
        Reorder mu, H, z, d by the values in mu.

        Parameters:
            mu (np.ndarray): p x K matrix
            H (np.ndarray): 1D array of length n
            z (np.ndarray): p x K matrix
            d (np.ndarray): 1D array of length K

        Returns:
            dict: Reordered versions of mu, H, z, d
        """
        
        mu = mu.copy()
        H = H.copy()
        d = d.copy()
        z = z.copy()

        K = mu.shape[1]
        p = mu.shape[0]
        n = len(H)

        stop = False
        l = np.arange(K)  # Equivalent to 1:K in R

        for i in range(p):
            if stop:
                break

            lc = l[np.argsort(mu[i, l])]

            # Temporary copies
            mutemp = mu.copy()
            Htemp = H.copy()
            dtemp = d.copy()
            ztemp = z.copy()

            for t, idx in enumerate(lc):
                mu[:, l[t]] = mutemp[:, idx]
                z[:, l[t]] = ztemp[:, idx]
                H[Htemp == idx] = l[t]
                d[l[t]] = dtemp[idx]

            l = np.sort(l[mu[i, l] == 0])
            if len(l) <= 1:
                stop = True

        return {"mu": mu, "H": H, "z": z, "d": d}

    def posterior_mean(self, attr: str):
        """Compute the posterior mean for each estimates.
        
        Parameters
        =========
        attr: attributes for estimation (e.g. mu, z, ...)
        """
        
        return self.posterior[attr].mean(axis = 0)
        
    
    def distance(self, vectors: np.ndarray, p: float = 1) -> np.ndarray:
        """
        Compute the pairwise Lp-norm distance matrix for a set of column vectors without explicit loops.

        Parameters:
            vectors (np.ndarray): A 2D array of shape (dim, N) where each column is a vector.
            p (float): The order of the norm (default is 2, Euclidean distance).

        Returns:
            np.ndarray: A 2D array of shape (N, N) containing pairwise distances.
        """
        # Expand dimensions for broadcasting
        vectors_expanded = vectors[:, :, np.newaxis]  # Shape (dim, N, 1)
        diffs = vectors_expanded - vectors_expanded.transpose(0, 2, 1)  # Shape (dim, N, N)

        # Compute Lp norm distances
        distance_matrix = np.linalg.norm(diffs, ord=p, axis=0)

        return distance_matrix
