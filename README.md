# Optimal k KMeans
A sklearn KMeans that automatically finds the optimal number of clusters based on the calinski_harabasz_score and the davies_bouldin_score. Great if the ground truth is not known.


## Usage:
Use inplace of any KMeans model.
```py
opt_model = OptimKMeans()
opt_model.fit(X)
preds = opt_model.predict()
```
can also be used in a pipeline:
```py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), KMeansOptimN())
pipe.fit(X)
preds = pipe.predict(X)
```
explore testing in testing.ipynb.

can also directly use the `get_optimal_clusters` function to find the optimal number of clusters for a KMeans estimator. 