# VariableSelectionModel
from .UnstructureVariableSelection import UnstructureVariableSelection
from .QualityCalculator import QualityCalculator

# Data Preprocessing Model
import numpy as np
import pandas as pd

# Visualization
import plotly.express as px

# Decomposition Models
from sklearn.decomposition import PCA
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecification

# Type Hinting
from typing import Union

class WordClusterAnalyzer(object):
    def __init__(
        self,
        model: UnstructureVariableSelection,
        seg_no = 0,
        k = 2,
        mode = 'PCA',
        cluster_label = None
    ):
        """Constructor of WordClusterAnalyzer. `UnstructureVariableSelection` model is needed (compatible module).
    
        Parameter
        ========
        model: fitted model from UnstructureVariableSelection
        k: number of clusters
        mode: dimension reduction model, either string or `factor analyzer` class from `sklearn.decompostition`. 
            + Compatible mode: PCA, CFA, Calculate
        """

        self.model = model
        self.seg_no = seg_no
        self.k = k
        self.mode = mode
        self.cluster_label = cluster_label

    def _fit_pca(self, frequency_matrix, cluster_id):
        return PCA(n_components = 1).fit(frequency_matrix.loc[:, cluster_id])
    
    def _fit_cfa(self, frequency_matrix, cluster_id):
        model_specified = ModelSpecification(
            np.array([[1] for _ in cluster_id]),
            n_factors = 1,
            n_variables = len(cluster_id)
        )
        return ConfirmatoryFactorAnalyzer(
            model_specified
        ).fit(frequency_matrix.loc[:, cluster_id])

    def _fit_cal(self, frequency_matrix, beta, cluster_id):
        return QualityCalculator().fit(
            frequency_matrix.loc[:, cluster_id], 
            beta.loc[cluster_id, 'coef']
        )
    
    def fit(self, cut = 0.95):
        k = self.k
        seg_no = self.seg_no

        if isinstance(self.cluster_label, dict):
            cluster_dict = self.cluster_label
        else:
            cluster_dict = self.model.find_group_members(seg_no, k, cut)
        fitter_dict = dict()
        
        for key in cluster_dict:
            cluster_id = cluster_dict[key]
            frequency_matrix = self.model.frequency_matrix
            if self.model.dist_type == 'correlate':
                frequency_matrix = frequency_matrix.div(self.model.weight, axis = 0)

            if self.mode == "PCA":
                fitter = self._fit_pca(frequency_matrix, cluster_id)
            elif self.mode == "CFA":
                fitter = self._fit_cfa(frequency_matrix, cluster_id)
            elif self.mode == "Calculate":
                coef = self.model.find_correlation_by_seg(seg_no, cut - 0.05)
                fitter = self._fit_cal(frequency_matrix, coef, cluster_id)
            else:
                fitter = self.mode(n_components = 1).fit(frequency_matrix.loc[:, cluster_id])
            
            fitter_dict[key] = fitter

        self.fitter_dict = fitter_dict
        self.cluster_dict = cluster_dict
    
        return self

    def transform(self, data_frame):
        if isinstance(data_frame, UnstructureVariableSelection):
            return self.transform_by_model(data_frame)
        
        value_dict = dict()
        
        for key in self.fitter_dict:
            fitter = self.fitter_dict[key]
            cluster_id = self.cluster_dict[key]

            data_col = data_frame.columns
            for col in cluster_id:
                if col not in data_col:
                    data_frame[col] = 0
            
            value = fitter.transform(data_frame.loc[:, cluster_id])
            value_dict[key] = value

        return value_dict


    def transform_by_model(self, model: UnstructureVariableSelection):
        """Need to combine with `self.transform()` function."""
        
        if self.model.dist_type == 'correlate':
            data_frame = model.frequency_matrix.div(model.weight, axis = 0)
        else:
            data_frame = model.frequency_matrix
        
        value_dict = dict()
        
        for key in self.fitter_dict:
            fitter = self.fitter_dict[key]
            cluster_id = self.cluster_dict[key]

            data_col = data_frame.columns
            for col in cluster_id:
                if col not in data_col:
                    data_frame[col] = 0
            
            value = fitter.transform(data_frame.loc[:, cluster_id])
            value_dict[key] = value

        reduction = pd.DataFrame(
            data = [
                value_dict[key].reshape(-1) for key in value_dict
            ] + [
                model.y,
                model.weight
            ],
            index = [f"cluster_{k}" for k in range(len(value_dict))] + ['rating', 'count'],
            columns = model.frequency_matrix.index
        ).transpose().reset_index()            

        return reduction











            