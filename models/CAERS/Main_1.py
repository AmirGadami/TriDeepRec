import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Load data
dot = pd.read_pickle('DOT.pkl')
labels = pd.read_pickle('LABELS.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dot, labels, test_size=0.2, random_state=122)

# Define custom loss function
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = y_true > 0  # Create a mask for non-zero values
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    squared_errors = tf.square(y_true_masked - y_pred_masked)
    mse = tf.reduce_mean(squared_errors)

    return mse


def RMSE(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = y_true > 0  # Create a mask for non-zero values

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    squared_errors = tf.square(y_true_masked - y_pred_masked)
    mse = tf.reduce_mean(squared_errors)
    rmse = tf.sqrt(mse)
    return rmse

def MAE(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = y_true > 0  # Create a mask for non-zero values

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    absolute_errors = tf.abs(y_true_masked - y_pred_masked)
    mae = tf.reduce_mean(absolute_errors)

    return mae


# Prepare 2D data
X_train_2d = X_train.values[:4800, :3000]
y_train_2d = y_train.values[:4800, :3000]
X_train_2d = X_train_2d.reshape((1, 4800, 3000, 1))
y_train_2d = y_train_2d.reshape((1, 4800, 3000, 1))

# Define 2D convolutional autoencoder model
def convolutional_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = decoded * 5

    # Create the autoencoder model
    autoencoder = Model(input_img, decoded)

    return autoencoder

# Create and compile the 2D convolutional autoencoder model
input_shape_2d = (4800, 3000, 1)
autoencoder_2d = convolutional_autoencoder(input_shape_2d)
autoencoder_2d.compile(optimizer='adam', loss=custom_loss, metrics=[RMSE, MAE])

# Print the model summary
autoencoder_2d.summary()

# Fit the model with 2D data
autoencoder_2d.fit(X_train_2d, y_train_2d, epochs=20, batch_size=32, shuffle=True)

# Prepare 1D data
input_shape_1d = (3000, 1)
X_train_1d = X_train.values[:4800, :3000].reshape((4800, 3000, 1))
y_train_1d = y_train.values[:4800, :3000].reshape((4800, 3000, 1))

# Define 1D convolutional autoencoder model
def convolutional_autoencoder_1d(input_shape):
    # Encoder
    input_seq = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(input_seq)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = Conv1D(32, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    decoded = decoded * 5

    # Create the autoencoder model
    autoencoder = Model(input_seq, decoded)

    return autoencoder

# Create and compile the 1D convolutional autoencoder model
autoencoder_1d = convolutional_autoencoder_1d(input_shape_1d)
autoencoder_1d.compile(optimizer='adam', loss=custom_loss, metrics=[RMSE, MAE])

# Print the model summary
autoencoder_1d.summary()

# Fit the model with 1D data
autoencoder_1d.fit(X_train_1d, y_train_1d, epochs=20, batch_size=32, shuffle=True)
