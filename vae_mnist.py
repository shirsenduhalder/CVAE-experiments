import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time 
import sys
import numpy as np
import glob2
import matplotlib.pyplot as plt
import PIL
import imageio
tfe = tf.contrib.eager
tf.enable_eager_execution()
from IPython import display

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
		self.inference_net.add(tf.keras.layers.Flatten())
		self.inference_net.add(tf.keras.layers.Dense(latent_dim + latent_dim))
		if use_BN:
			self.inference_net.add(tf.keras.layers.BatchNormalization())

		self.generative_net = tf.keras.Sequential()
		self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(latent_dim, )))
		self.generative_net.add(tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu))
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
		if eps is None:
			eps = tf.random_normal(shape=(100, self.latent_dim))

		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)

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

#LOSS FUNCTIONS
def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.log(2. * np.pi)

	return tf.reduce_sum(-0.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def compute_loss(model, x):
	mean, logvar = model.encode(x)
	z = model.reparametrize(mean, logvar)
	decoded_x = model.decode(z)

	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded_x, labels=x)

	logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
	logpz = log_normal_pdf(z, 0., 0.)
	logqz_x = log_normal_pdf(z, mean, logvar)

	return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)

	return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)

def apply_gradients(optimizer, gradients, variables, global_step=None):
	optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


epochs = 100
latent_dim = 50
num_examples = 16

random_vector = tf.random_normal(shape=[num_examples, latent_dim])
model = CVAE(latent_dim)

def save_images(model, epoch, test_input, use_BN=False):
	predictions = model.sample(test_input)
	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, (i + 1))
		plt.imshow(predictions[i, :, :, 0], cmap='gray')
		plt.axis('off')

	savedir = 'BatchNorm' if use_BN else 'NoBatchNorm'

	if not os.path.exists(savedir):
		os.makedirs(savedir)

	plt.savefig(os.path.join(savedir, 'image_at_epoch_{:04d}.png'.format(epoch)))

save_images(model, 0, random_vector)

for epoch in range(1, epochs + 1):
	start_time = time.time()
	for train_batch in train_dataset:
		gradients, loss = compute_gradients(model, train_batch)
		apply_gradients(optimizer, gradients, model.trainable_variables)
	end_time = time.time()

	if epoch%1 == 0:
		loss = tfe.metrics.Mean()
		for test_batch in test_dataset:
			loss(compute_loss(model, test_batch))
		elbo = -loss.result()
		display.clear_output(wait=False)

		print("Epoch: {}, Test set ELBO: {}, time taken: {}".format(epoch, elbo, (end_time - start_time)))
		save_images(model, epoch, random_vector)


def display_image_epoch(epoch_no, use_BN):
	savedir = 'BatchNorm' if use_BN else 'NoBatchNorm'
	assert os.path.exists(savedir), "No such folder"

	return PIL.Image.open(os.path.join(savedir, 'image_at_epoch_{:04d}.png'.format(epoch_no)))

display_image_epoch(epochs)