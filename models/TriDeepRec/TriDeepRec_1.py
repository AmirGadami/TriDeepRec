import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load data
allData = pd.read_pickle('allData_1.pkl')
allData.dropna(inplace=True)

# Define columns
user_c = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
rating_c = ['user_id', 'movie_id', 'rating', 'time']
movie_c = ['movie_id', 'movie_name', 'genre']

# Split data
movie = allData[movie_c]
user = allData[user_c]
rating = allData[rating_c]
rating.columns = ["user", "item", "label", "time"]

# Prepare train, evaluation, and test data
train_data, eval_data, test_data = random_split(rating, multi_ratios=[0.8, 0.1, 0.1])
train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)

# Clear previous sessions
tf.keras.backend.clear_session()

# Define and fit NCF model with expanded hyperparameter tuning
param_dist_ncf = {
    'embed_size': [64, 128, 256],  # Including larger sizes for exploration
    'lr': [0.001, 0.01, 0.1],      # Wider range of learning rates
    'batch_size': [32, 64, 128],   # Larger batch sizes for exploration
    'dropout_rate': [0.0, 0.5, 0.7], # Additional dropout rate for exploration
    'hidden_units': [(64, 128), (128, 256), (256, 512)]  # More varied architectures
}

ncf = NCF(task="rating", data_info=data_info, loss_type="mse", n_epochs=3, num_neg=1)
random_search_ncf = RandomizedSearchCV(estimator=ncf, param_distributions=param_dist_ncf, n_iter=10, cv=3, verbose=2)
random_search_ncf.fit(train_data)

# Evaluate the best NCF model
best_ncf = random_search_ncf.best_estimator_
evaluate(model=best_ncf, data=test_data, metrics=["rmse", "mae"])

# Prediction
rating['pred'] = best_ncf.predict(rating['user'], rating['item'])

# Load predictions and merge data
caers_pred = pd.read_pickle('predictions_2.pkl')
pred_rating = rating.pivot(index='user', columns='item', values='pred')
y_true = rating.pivot(index='user', columns='item', values='label')

# Prepare data for MLP with expanded hyperparameter tuning
caers = caers_pred.reset_index().melt(id_vars='user_id', var_name='movie_id', value_name='caers_rating')
caers.columns = ['user', 'item', 'caers_rating']
merged_data = pred_rating.merge(y_true, on=['user', 'item']).merge(caers, on=['user', 'item'])
merge_data = merged_data[merged_data['caers_rating'] != 0]

X = merge_data.loc[:, ['pred', 'caers_rating']].values
y = merge_data['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

param_dist_mlp = {
    'hidden_layer_sizes': [(32,), (64,), (128,)],  # Expanded hidden layer sizes
    'alpha': [0.0001, 0.001, 0.01]  # Wider range of regularization strengths
}

mlp = MLPRegressor(max_iter=1000, random_state=4, early_stopping=True)
random_search_mlp = RandomizedSearchCV(estimator=mlp, param_distributions=param_dist_mlp, n_iter=10, cv=3, verbose=2)
random_search_mlp.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = random_search_mlp.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE: {:.4f}".format(mse))
