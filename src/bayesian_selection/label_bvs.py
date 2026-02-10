import pandas as pd
import numpy as np
from .bvs import BayesianVariableSelection

class LabelBVS(BayesianVariableSelection):
    def __init__(self, X, y, stop_word_path = None, weight = None, dist_type = 'cooccur', track = False):
        """Constructor of LabelBVS
        
        `LabelBVS` supports auto label procedure during topic modeling based on Kim, Lee, and McCulloch (2024, Journal of Marketing Research)
        The word label is automatically saved from X.columns
        Note that the latent regression admits intercept
        """
        if stop_word_path != None:
            X = self.filter_words(X, stop_word_path)
        if isinstance(weight, type(None)) == False:
            weight = weight.to_numpy().reshape(-1)
            
        self.word_label = X.columns
            
        X = np.concatenate(
            [
                np.ones(shape = (len(X), 1)), 
                X.to_numpy()
            ],
            axis = 1
        )
        
        super().__init__(X, y.to_numpy().reshape(-1), weight, dist_type, track)
    
    def diagnose_model(self, sig = 0.05):
        result = super().diagnose_model(sig = sig)
        result['variable'] = result['variable'].apply(lambda x: self.word_label[x - 1])
        
        return result
    
    def filter_words(self, X, stop_word_path):
        stop_word = pd.read_excel(stop_word_path)
        stop_word.columns = ['valid_corpus', 'change']
        
        stop_word = stop_word.dropna(subset = ['valid_corpus'])
        stop_word['change'] = stop_word['change'].fillna(stop_word['valid_corpus'])
        
        stop_word = stop_word.set_index('valid_corpus')

        X_t = X.transpose()
        X_t = X_t.join(stop_word, how = 'inner')
        
        X_t = X_t.groupby('change').sum()
        
        return X_t.transpose()
    
        
    def find_topic_by_seg(self, k, p_dim, cut = 0.95):
        symm_M0, sel_t = super().find_topic_by_seg(k, p_dim, cut)
        sel_word = self.word_label[sel_t]
        
        return symm_M0, sel_word
    
    def find_correlation_by_seg(self, k, cut = 0.95):
        df = super().find_correlation_by_seg(k, cut)
        df.index = self.word_label
        
        return df.query(f"select_ratio > {cut}")
    
    def export_data(self) -> pd.DataFrame:
        X, y = super().export_data()
        X.columns = ['Intercept'] + list(self.word_label)
        return (X, y)

    def find_group_members(self, seg_no, k = 2, cut = 0.95):
        cluster_dict = super().find_group_members(seg_no, k, cut)
        
        for value in cluster_dict:
            cluster_dict[value] = self.word_label[self.select_ind][np.array(cluster_dict[value], dtype = 'int')]

        return cluster_dict

    