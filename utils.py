import numpy as np
import pandas as pd
import statsmodels.api as sm

# Preprocessing Tool
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

# Visualization
import plotly.express as px

MODEL_METRIC_DEFAULT = ['Dep. Variable', 'Model', 'No. Observations', 'Log-Likelihood', 'Pseudo R-squ.']
MODEL_METRIC_OLS = ['Dep. Variable', 'Model', 'No. Observations', 'R-squared']

def descriptive_statistic(data, numeric_columns, category_columns, quantile_list = [0, .25, .5, .75, 1], quantile_label = ['min', '1Q', 'med', '3Q', 'max']):
    return numeric_statistic(data, numeric_columns, quantile_list, quantile_label), categoric_statistic(data, category_columns)

def numeric_statistic_by_row(data: pd.DataFrame, column: str, index: int, quantile_list: list = [0, .25, .5, .75, 1], quantile_label: list = ['min', '1Q', 'med', '3Q', 'max']):
    """Derive Numeric Statistic for a single variable
    
    Parameter
    ==========
    data: pd.DataFrame
    column: str
    index: int
    quantile_list: list
    quantile_label: list
    """
    if len(quantile_list) != len(quantile_label):
        raise ValueError(f"length of quantile list ({len(quantile_list)}) and label ({len(quantile_label)}) are different")
    
    numeric_data = data[column]
    df_row = pd.Series([
            index,
            column,
            'numeric',
            numeric_data.count(),
            numeric_data.isna().sum(),
            numeric_data.mean(),
            numeric_data.std(),
        ] + [numeric_data.quantile(q) for q in quantile_list],
        index = ['index', 'var_name', 'var_type', 'count', 'null', 'mean', 'std'] + quantile_label
    )
    
    return df_row

def categoric_statistic_by_row(data, column, index, level = 5):
    """Derive Categoric Statistic for a single variable
    
    Parameter
    ==========
    data: pd.DataFrame
    column: str
    index: int
    """
    categoric_data = data[column]
    basic_info = pd.Series([index, column, 'categoric',
                            categoric_data.count(),
                            categoric_data.isna().sum(),
                            categoric_data.nunique()
                           ], index = ['index', 'var_name', 'var_type', 'count', 'null', 'nunique'])
    content_info = categoric_data.value_counts()[:5].apply(lambda x: f"{x:,}")
    content_info = pd.Series(content_info)
    
    try:
        content_info = content_info.apply(lambda x: f"{x:,}")
    except:
        pass
    content_info = pd.Series(content_info.index, index = content_info.index).apply(str) + " (" + content_info + ")"
    
    content_info.index = [f'level {i + 1}' for i in range(content_info.count())]
    content_info = content_info[:min(content_info.count(), level)]
    
    df_row = pd.concat([basic_info, content_info])
    return df_row

def numeric_statistic(data, columns, quantile_list, quantile_label):
    return pd.DataFrame([numeric_statistic_by_row(data, col, index + 1, quantile_list, quantile_label) for index, col in enumerate(columns)])

def categoric_statistic(data, columns):
    return pd.DataFrame([categoric_statistic_by_row(data, col, index + 1) for index, col in enumerate(columns)])

def summary_to_df(summary_obj, metrics = MODEL_METRIC_DEFAULT, parenthesis = 'std_err', significant = 0.05):
    """Make readable dataframe from Summary Object in `statsmodels.api`
    
    Parameter
    =========
    summary_obj: statsmodels.api summary object
    
    Return
    ======
    summary text dataframe with coefficient and report table
    
    Example
    =======
    | variable | coef |
    |    X1    |  100 | 
    |    X2    |  120 | 
    |   Rsq    |  0.8 | 
    """
    
    
    metric_df = summary_to_df_metric(summary_obj.tables[0], metrics)
    param_df = summary_to_df_params(summary_obj.tables[1], parenthesis, significant)
    
    df = pd.concat([param_df, metric_df]).set_index('variable')
    df.columns = df.loc['Dep. Variable', :]
    df = df.drop(index = 'Dep. Variable')
    
    return df

def summary_to_df_metric(metric_summary, metrics):
    model_metric = pd.DataFrame(metric_summary.data)
    model_metric_p1 = model_metric.iloc[:, :2]
    model_metric_p2 = model_metric.iloc[:, 2:]

    model_metric_p1.columns = ['variable', 'value']
    model_metric_p2.columns = ['variable', 'value']

    model_metric_p2['variable'] = model_metric_p2['variable'].apply(lambda x: x[2:])
    model_metric = pd.concat([model_metric_p1, model_metric_p2])
    model_metric['variable'] = model_metric['variable'].apply(lambda x: x.split(':')[0])
    model_metric = model_metric.set_index('variable')
    return model_metric.loc[metrics, :].reset_index()

def summary_to_df_params(parameter_summary, parenthesis = 'std_err', significant = 0.05):
    model_param = pd.DataFrame(parameter_summary.data).iloc[1:, :]
    model_param.columns = ['variable', 'parameter', 'std_err', 't_stat', 'p_value', 'CI_low', 'CI_upp']
    
    model_param['p_value'] = model_param['p_value'].apply(lambda x: float(x))
    
    if significant == 0.05:
        model_param['value'] = model_param['parameter'] +  np.where(model_param['p_value'] < 0.001, '***', 
                                                                    np.where(model_param['p_value'] < 0.01, '**',
                                                                            np.where(model_param['p_value'] < 0.05, '*', '')
                                                                            )
                                                                   ) + "\n(" + model_param[parenthesis].astype(str).replace(' ', '')  + ")"
    elif significant == 0.1:
        model_param['value'] = model_param['parameter'] +  np.where(model_param['p_value'] < 0.01, '***', 
                                                                    np.where(model_param['p_value'] < 0.05, '**',
                                                                            np.where(model_param['p_value'] < 0.1, '*', '')
                                                                            )
                                                                   ) + "\n(" + model_param[parenthesis].astype(str).replace(' ', '')  + ")"
    else:
        raise ValueError("Incompatible Signficance. Please Check significant argument")
    return model_param[['variable', 'value']]


def find_cluster_members(linkage, cluster_no, cluster_list = None):
    """Find cluster member from scipy.cluster.hierarchy.linkage.
    
    Parameter
    =========
    linkage: linkage matrix generated from scipy.cluster.hierarchy.linkage
    cluster_no: index of cluster (sample_count + 1 ~ 2 * sample_count)
    """
    if cluster_list == None:
        cluster_list = list()
        
    elem_no = len(linkage) + 1
    cluster_index = cluster_no - elem_no
    if cluster_index < 0:
        return [cluster_no]
        
    left_index, right_index, _, _ = linkage[int(cluster_index)]

    if left_index < elem_no:
        cluster_list.append(left_index)
    else:
        find_cluster_members(linkage, left_index, cluster_list)

    if right_index < elem_no:
        cluster_list.append(right_index)
    else:
        find_cluster_members(linkage, right_index, cluster_list)
    
    return cluster_list

def apply_lof(df, latent_axis = ['cluster_0', 'cluster_1'], n_neighbors = 20):
    lof = LocalOutlierFactor(n_neighbors = n_neighbors, p = 2, metric = 'euclidean')
    y_pred = lof.fit_predict(
        StandardScaler().fit_transform(
        df[latent_axis])
    )
    # df['outlier'] = np.clip(lof.negative_outlier_factor_, -5, 0)
    df['LOF'] = y_pred.astype(str)
    return df

def apply_dbscan(df, latent_axis = ['cluster_0', 'cluster_1'], eps = 0.5):
    dbscan = DBSCAN(eps = eps, p = 2, metric = 'euclidean')
    y_pred = dbscan.fit_predict(
        StandardScaler().fit_transform(
        df[latent_axis])
    )
    df['DBSCAN'] = y_pred
    df['DBSCAN'] = df['DBSCAN'].astype(str)
    return df

def apply_heuristic(df, latent_axis = ['cluster_0', 'cluster_1'], sigma_rule = 3):
    data = StandardScaler().fit_transform(df[latent_axis])
    df['Heuristic'] = (data ** 2).sum(axis = 1) > (sigma_rule ** 2)
    df['Heuristic'] = df['Heuristic'].replace(True, -1).replace(False, 1)
    df['Heuristic'] = df['Heuristic'].astype(str)
    return df

def apply_mcd(df, latent_axis = ['cluster_0', 'cluster_1']):
    mcd = MinCovDet().fit(df[latent_axis])
    robust_mahalanobis = mcd.mahalanobis(df[latent_axis])
    # IQR
    q1 = np.percentile(robust_mahalanobis, 25)
    q3 = np.percentile(robust_mahalanobis, 75)
    iqr = (q3 - q1)
    
    lower_bound = q1 - iqr * 3
    upper_bound = q3 + iqr * 3
    
    df['MCD'] = ((robust_mahalanobis > upper_bound) | (robust_mahalanobis < lower_bound)) * 1
    df['MCD'] = (-df['MCD']).astype(str)

    return df

def detect_outlier(data_frame, groupby_keys = None, latent_axis = ['cluster_0', 'cluster_1'], lof_dict = {}):
    if groupby_keys == None:
        groupby_keys = '_const'
        data_frame[groupby_keys] = 1
    data_frame = data_frame.groupby(groupby_keys, group_keys = False, sort = False).apply(lambda x: apply_lof(x, latent_axis, **lof_dict))
    data_frame = data_frame.groupby(groupby_keys, group_keys = False, sort = False).apply(apply_dbscan)
    data_frame = data_frame.groupby(groupby_keys, group_keys = False, sort = False).apply(apply_heuristic)
    data_frame = data_frame.groupby(groupby_keys, group_keys = False, sort = False).apply(apply_mcd)
    
    return data_frame

def visualize_scatter_plot(data_frame, id_vars, latent_axis = ['cluster_0', 'cluster_1'], values = ['LOF', 'Heuristic', 'DBSCAN', 'MCD'], **kwarg):
    data_frame = data_frame.melt(id_vars = id_vars + latent_axis, value_vars = values)
    data_frame['value'] = data_frame['value'].astype(str)
    data_frame = data_frame.sort_values(id_vars + ['value'])
    
    return px.scatter(
        data_frame = data_frame,
        x = latent_axis[0],
        y = latent_axis[1],
        color = 'value',
        facet_row = 'variable',
        template = 'simple_white',
        **kwarg
    )

def visualize_outlier_plot(data_frame, id_vars, latent_axis = ['cluster_0', 'cluster_1'], values = ['LOF', 'Heuristic', 'DBSCAN', 'MCD'], **kwarg):
    data_frame = data_frame.melt(id_vars = id_vars + latent_axis, value_vars = values)
    data_frame['value'] = data_frame['value'].astype(str)
    data_frame = data_frame.sort_values(id_vars + ['value'])
    data_frame = data_frame[data_frame['value'] == "-1"]
    
    return px.scatter(
        data_frame = data_frame,
        x = latent_axis[0],
        y = latent_axis[1],
        facet_row = 'variable',
        color = 'rating',
        template = 'simple_white',
        **kwarg
    )

def summarize_outlier(data_frame, id_vars, latent_axis = ['cluster_0', 'cluster_1'], values = ['LOF', 'Heuristic', 'DBSCAN', 'MCD']):
    data_frame = data_frame.melt(id_vars = id_vars + latent_axis, value_vars = values)
    data_frame['value'] = data_frame['value'].astype(int)
    data_frame['isoutlier'] = (data_frame['value'] == -1) * 1
    return data_frame.pivot_table(index = 'variable', columns = id_vars, values = 'isoutlier', aggfunc = 'mean')

def construct_outlier_pivot_table(data_frame, key, detect_methods, values, aggfunc):
    df_list = list()
    
    for method in detect_methods:
        df_list.append(
            construct_outlier_single_pivot_table(data_frame, key, method, values, aggfunc)
        )
    return pd.concat(df_list, ignore_index = True)

def construct_outlier_single_pivot_table(data_frame, key, detect_method, values, aggfunc):
    data_frame[f"{detect_method}_outlier"] = data_frame[detect_method].apply(lambda x: -1 if x == '-1' else 1)    
    pivot_table = data_frame.pivot_table(columns = key, index = f"{detect_method}_outlier", values = values, aggfunc = aggfunc).reset_index().rename(
        columns = {
            f"{detect_method}_outlier": 'outlier'
        }
    )
    pivot_table['method'] = detect_method
    
    return pivot_table