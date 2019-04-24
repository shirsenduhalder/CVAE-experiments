import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()
import time 
import sys
import numpy as np
import glob2
import matplotlib.pyplot as plt
import PIL
import imageio

#LOADING DATA
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32')

train_images = train_images/255.0
test_images = test_images/255.0

train_images[train_images >= 0.5] = 1.
train_images[train_images <= 0.5] = 0.
test_images[test_images >= 0.5] = 1.
test_images[test_images <= 0.4] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 200

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

#CREATING THE CVAE MODEL
class CVAE(tf.keras.Model):
	def __init__(self, latent_dim, use_BN=False):
		super(CVAE, self).__init__()
		self.latent_dim = latent_dim

		self.inference_net = tf.keras.Sequential()
		self.inference_net.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
		self.inference_net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(3, 3), activation=tf.nn.relu))
		if use_BN:
			self.inference_net.add(tf.keras.layers.BatchNormalization())
		self.inference_net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(3, 3), activation=tf.nn.relu))
		if use_BN:
			self.inference_net.add(tf.keras.layers.BatchNormalization())
		self.inference_net.add(tf.keras.layers.flatten())
		self.inference_net.add(tf.keras.layers.Dense(latent_dim + latent_dim))
		if use_BN:
			self.inference_net.add(tf.keras.layers.BatchNormalization())

		self.generative_net = tf.keras.Sequential()
		self.generative_net.add(tf.keras.InputLayer(input_shape=(latent_dim, )))
		self.generative_net.add(tf.keras.Dense(units=7*7*32), activation=tf.nn.relu)
		if use_BN:
			self.generative_net.add(tf.keras.layers.BatchNormalization())
		self.generative_net.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
		self.generative_net.add(tf.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.relu))
		if use_BN:	
			self.generative_net.add(tf.keras.layers.BatchNormalization())
		self.generative_net.add(tf.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation=tf.nn.relu))
		if use_BN:
			self.generative_net.add(tf.keras.layers.BatchNormalization())
		self.generative_net.add(tf.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME"))

	def sample(self, eps=None):
		if eps is not None:
			eps = tf.random_normal(shape=(100, self.latent_dim))

		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self, inference_net(x), num_or_size_splits=2, axis=1)

		return mean, logvar

	def reparametrize(self, mean, logvar):
		eps = tf.random_normal(shape=mean.shape)

		return eps * tf.exp(logvar * 0.5) + mean

	def decode(self, z, apply_sigmoid=False):
		logits = self.generative_net(z)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)

			return probs

		return logits