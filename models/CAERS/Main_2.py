import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Load data
dot = pd.read_pickle('DOT100k.pkl')
labels = pd.read_pickle('LABELS100k.pkl')

# Define custom loss functions
def custom_loss(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, y_true > 0)
    y_pred_masked = tf.boolean_mask(y_pred, y_true > 0)
    
    squared_errors = tf.square(y_true_masked - y_pred_masked)
    mse = tf.reduce_mean(squared_errors)
    
    return mse

def RMSE(y_true, y_pred):
    mse = custom_loss(y_true, y_pred)
    rmse = tf.sqrt(mse)
    
    return rmse

def MAE(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, y_true > 0)
    y_pred_masked = tf.boolean_mask(y_pred, y_true > 0)

    absolute_errors = tf.abs(y_true_masked - y_pred_masked)
    mae = tf.reduce_mean(absolute_errors)
    
    return mae

# Prepare data
X = dot.values[:780, :48]
Y = labels.values[:780, :48]
X_train = X.reshape((1, -1, 48, 1))
y_train = Y.reshape((1, -1, 48, 1))

# Create the convolutional autoencoder model
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

# Assuming your data shape is (780, 48, 1)
input_shape = (780, 48, 1)

# Create the convolutional autoencoder model
autoencoder = convolutional_autoencoder(input_shape)

# Print the model summary
autoencoder.summary()

# Compile and fit the model
autoencoder.compile(optimizer='adam', loss=custom_loss, metrics=[RMSE, MAE])
history = autoencoder.fit(X_train, y_train, epochs=20, batch_size=32, shuffle=True)
