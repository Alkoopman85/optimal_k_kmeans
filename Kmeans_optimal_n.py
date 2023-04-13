from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_is_fitted


from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

import numpy as np

from typing import Tuple, List



def get_optimal_clusters(km_model:KMeans, X, measure:str='comb', cluster_range:tuple=(2, 10, 1)) -> Tuple[int, List[float], List[float]]:
    """Calculates the optimal number of clusters given a KMeans model, 
        based on the calinski harabasz score and or the davies bouldin score.

    Args:
        km_model (KMeans): an istance of sklearn KMeans
        X (np.array, DataFrame): input data to Kmeans algorithm
        measure (str, optional): metric for optimizing the number of clusters can be "comb", "chs", "dbs".
                chs = calinski_harabasz_score, dbs = davies_bouldin_score, comb is a combination. Defaults to 'comb'.
        cluster_range (tuple, optional): cluster range to search. Defaults to (2, 10, 1).

    Returns:
        Tuple[int, List[float], List[float]]: (the optimal number of clusters, calinski harabasz scores, davies bouldin scores) 
    """

    chs_scores = []
    dbs_scores = []
    for n_cluster in range(cluster_range[0], cluster_range[1], cluster_range[2]):
        model_ = clone(km_model)
        model_.set_params(n_clusters=n_cluster)
        model_.fit(X)
        labels = model_.labels_

        if measure == 'chs' or measure == 'comb':
            chs_score = calinski_harabasz_score(X, labels)
            chs_scores.append((n_cluster, chs_score))

        if measure == 'dbs' or measure == 'comb':
            dbs_score = davies_bouldin_score(X, labels)
            dbs_scores.append((n_cluster, dbs_score))

    if measure == 'chs':
        return max(chs_scores, key=lambda x: x[1])[0], chs_scores, dbs_score

    elif measure == 'dbs':
        return min(dbs_scores, key=lambda x: x[1])[0], chs_scores, dbs_score

    else:
        max_chs_score = max(chs_scores, key=lambda x: x[1])
        min_dbs_score = min(dbs_scores, key=lambda x: x[1])
        return round(np.mean([max_chs_score[0], min_dbs_score[0]])), chs_scores, dbs_scores


class KMeansOptimN(ClusterMixin, BaseEstimator):
    """Kmeans class that has the option to automatically find the optimal number of clusters based on the
        calinski_harabasz_score, davies_bouldin_score, or a combination
    """
    def __init__(self, n_clusters:int|None=None, init:str='k-means++', 
                n_init:str|int='auto', max_iter:int=300, tol:float=1e-4, 
                verbose:int=0, random_state:int|None=None, copy_x:bool=True,
                algorithm:str='lloyd') -> None:
        """

        Args:
            n_clusters (int | None, optional): if None the optimal number will be found. Defaults to None.
            init (str, optional): argument passed to sklearn KMeans. Defaults to 'k-means++'.
            n_init (str | int, optional): argument passed to for sklearn KMeans. Defaults to 'auto'.
            max_iter (int, optional): argument passed to for sklearn KMeans. Defaults to 300.
            tol (float, optional): argument passed to for sklearn KMeans. Defaults to 1e-4.
            verbose (int, optional): argument passed to for sklearn KMeans. Defaults to 0.
            random_state (int | None, optional): argument passed to for sklearn KMeans. Defaults to None.
            copy_x (bool, optional): argument passed to for sklearn KMeans. Defaults to True.
            algorithm (str, optional): argument passed to for sklearn KMeans. Defaults to 'lloyd'.
        """

        super(KMeansOptimN, self).__init__()
        self.n_clusters = n_clusters

        self.KMeans = KMeans(init=init, n_init=n_init, max_iter=max_iter,
                            tol=tol, verbose=verbose, random_state=random_state,
                            copy_x=copy_x, algorithm=algorithm)

    
    def get_optimal_clusters(self, X, measure:str='comb', cluster_range:tuple=(2, 10, 1)) -> Tuple[int, List[float], List[float]]:
        """Calculates the optimal number of clusters given a KMeans model, 
            based on the calinski harabasz score and or the davies bouldin score. 

        Args:
            X (np.array, DataFrame): input data to Kmeans algorithm
            measure (str, optional): metric for optimizing the number of clusters can be "comb", "chs", "dbs".
                chs = calinski_harabasz_score, dbs = davies_bouldin_score, comb is a combination. Defaults to 'comb'.
            cluster_range (tuple, optional): cluster range to search. Defaults to (2, 10, 1).

        Returns:
            Tuple[int, List[float], List[float]]: (the optimal number of clusters, calinski harabasz scores, davies bouldin scores)
        """
        optim_clusters = get_optimal_clusters(km_model=self.KMeans, X=X, measure=measure, cluster_range=cluster_range)
        return optim_clusters


    def fit(self, X, y=None, measure:str='comb', cluster_range:tuple=(2, 10, 1)):
        """fits the sklearn KMeans model. if self.n_clusters is None then the optimal number of clusters is calculated

        Args:
            X (np.array, DataFrame): data to fit
            y (optional): ignored. Defaults to None.
            measure (str, optional): metric for optimizing the number of clusters can be "comb", "chs", "dbs".
                chs = calinski_harabasz_score, dbs = davies_bouldin_score, comb is a combination. Defaults to 'comb'.
            cluster_range (tuple, optional): cluster range to search. Defaults to (2, 10, 1).

        Raises:
            Exception: if self.n_clusters is not None or int raises exception
        """
        if isinstance(self.n_clusters, int):
            self.KMeans.set_params(**{
                'n_clusters': self.n_clusters
            })

        elif self.n_clusters is None:
            n_clusters, _, _ = self.get_optimal_clusters(X, measure=measure, cluster_range=cluster_range)
            self.KMeans.set_params(**{
                'n_clusters': n_clusters
            })
        else:
            raise Exception()
        
        self.KMeans.fit(X)
        return self

    def predict(self, X, y=None):
        """predicts cluster assignments

        Args:
            X (np.array, DataFrame): data to predict
            y (optional): ignored. Defaults to None.

        Returns:
            ndarray: cluster assignments
        """

        check_is_fitted(self.KMeans)
        clust_labels = self.KMeans.predict(X)
        return clust_labels

    def fit_predict(self, X, y=None, measure:str='comb', cluster_range:tuple=(2, 10, 1)):
        """_summary_

        Args:
            X (np.array, DataFrame): data to fit and predict
            y (optional): ignored. Defaults to None.
            measure (str, optional): metric for optimizing the number of clusters can be "comb", "chs", "dbs".
                chs = calinski_harabasz_score, dbs = davies_bouldin_score, comb is a combination. Defaults to 'comb'.
            cluster_range (tuple, optional): cluster range to search. Defaults to (2, 10, 1).

        Returns:
            ndarray: cluster assignments
        """
        preds = self.fit(X, measure=measure, cluster_range=cluster_range).predict(X)
        return preds
