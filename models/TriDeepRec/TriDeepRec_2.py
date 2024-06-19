import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate

# Load data
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_100k = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies_100k = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users_100k = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# Prepare train, evaluation, and test data
train_data, eval_data, test_data = random_split(ratings_100k, multi_ratios=[0.8, 0.1, 0.1])
train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)

# Clear previous sessions
tf.keras.backend.clear_session()

# Set seed
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define NCF model with hyperparameter tuning
param_dist_ncf = {
    'embed_size': [64, 128, 256],  # Including larger sizes for exploration
    'lr': [0.001, 0.01, 0.1],      # Wider range of learning rates
    'batch_size': [32, 64, 128],   # Larger batch sizes for exploration
    'dropout_rate': [0.0, 0.5, 0.7], # Additional dropout rate for exploration
    'hidden_units': [(64, 128, 512), (128, 256, 512), (256, 512, 1024)]  # More varied architectures
}

ncf = NCF(task="rating", data_info=data_info, loss_type="mse", n_epochs=6, num_neg=1)
random_search_ncf = RandomizedSearchCV(estimator=ncf, param_distributions=param_dist_ncf, n_iter=10, cv=3, verbose=2, random_state=seed_value)
random_search_ncf.fit(train_data)

# Evaluate the best NCF model
best_ncf = random_search_ncf.best_estimator_
evaluate(model=best_ncf, data=test_data, metrics=["rmse", "mae"])

# Load additional data
allData = pd.read_pickle('Data/allData.pkl')

# Predict and merge data for evaluation
rating = allData[['user_id', 'movie_id', 'rating']]
rating['pred'] = best_ncf.predict(rating['user_id'], rating['movie_id'])

y_true = rating.pivot(index='user_id', columns='movie_id', values='rating')
pred_rating = rating.pivot(index='user_id', columns='movie_id', values='pred')

caers_pred = pd.read_pickle('predictions.pkl')
caers = caers_pred.reset_index().melt(id_vars='user_id', var_name='movie_id', value_name='caers_rating')
pred_rating = pred_rating.iloc[:780, :]
y_true = y_true.iloc[:780, :]
merged_data = pred_rating.merge(y_true, on=['user_id', 'movie_id']).merge(caers, on=['user_id', 'movie_id'])
merged_data = merged_data[merged_data['caers_rating'] != 0]

# Prepare data for MLP with hyperparameter tuning
X = merged_data[['pred', 'caers_rating']].values
y = merged_data['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

param_dist_mlp = {
    'hidden_layer_sizes': [(32,), (64,), (128,)],  # Expanded hidden layer sizes
    'alpha': [0.0001, 0.001, 0.01]  # Wider range of regularization strengths
}

mlp = MLPRegressor(max_iter=1000, random_state=4, early_stopping=True)
random_search_mlp = RandomizedSearchCV(estimator=mlp, param_distributions=param_dist_mlp, n_iter=10, cv=3, verbose=2, random_state=seed_value)
random_search_mlp.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = random_search_mlp.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE: {:.4f}".format(mse))
