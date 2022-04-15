# grid search class weights with svm for imbalance classification
from numpy import mean
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

# ifty
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)

# define model
model = SVC(gamma='scale')
# define grid
balance = [{0: 100, 1: 1}, {0: 10, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 100}]
param_grid = dict(class_weight=balance)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
# grid_result = grid.fit(X, y)

grid_result = grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# report the best configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # report all configurations
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
