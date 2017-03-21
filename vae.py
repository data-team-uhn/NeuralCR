import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
import fasttext_reader as reader
import h5py
import sys
import phraseConfig
#import gpu_access


def linear(name, x, shape):
	w = weight_variable(name + 'W', shape)
	b = weight_variable(name + 'B',(shape[1]))
	return tf.matmul(x,w) + b

def kernel_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
	#return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev = 0.02))

class Config:
        hpo_size = 11442
	hidden_size = 1024
	fc_size = 1024
	read_size = 100
	z_dim = 16
        learning_rate = 0.001
	lr_decay = 0.95
	batch_size = 11442

class NGA:
	def __init__(self, config):
		self.config = config
		self.x = tf.placeholder(tf.float32, shape = [None, 2*config.hpo_size])
		self.eps = tf.placeholder(tf.float32, shape = [None, config.z_dim])

                '''
		self.eps = tf.random_normal((config.batch_size, config.z_dim), 0, 1, 
				dtype=tf.float32)
                '''

		self.z_mean, self.z_log_sigma_sq = self.encoder()

		self.z = tf.add(self.z_mean, 
				(tf.sqrt(tf.exp(self.z_log_sigma_sq)) * self.eps))

		self.x_recon_theta = self.decoder()

		self.create_loss()

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.run(tf.initialize_all_variables())


	def encoder(self):
		# conv1
		# x.shape = [config.batch_size, 1, config.read_size, 4]

		with tf.variable_scope('enc_fc') as scope:
                    fc_layer1 = tf.nn.tanh(linear('w1', self.x, [2*self.config.hpo_size, self.config.hidden_size]))
                    fc_layer2 = tf.nn.tanh(linear('w2', fc_layer1, [self.config.hidden_size, self.config.fc_size]))
                    z_mean = linear('mean', fc_layer2, (self.config.fc_size, self.config.z_dim))
                    z_log_sigma_sq = linear('sigma', fc_layer2, (self.config.fc_size, self.config.z_dim))

		return (z_mean, z_log_sigma_sq)


	def decoder(self):
            
            with tf.variable_scope('dec_fc') as scope:
                init_state_fc1 = tf.nn.tanh(linear('w1', self.z, [self.config.z_dim, self.config.fc_size]))
                init_state_fc2 = tf.nn.tanh(linear('w2', init_state_fc1, [self.config.fc_size, self.config.hidden_size]))
                final_state = tf.nn.sigmoid(linear('w3', init_state_fc2, [self.config.hidden_size, 2*self.config.hpo_size]))

            return final_state

	def reconstr_loss(self):
            return -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_recon_theta) +\
                                    (1-self.x)*tf.log(1e-10 +1.0 - self.x_recon_theta), 1)


	def create_loss(self):
		'''
		self.reconstr_loss = \
				-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_recon_theta) ,[1,2,3])
		'''
		self.recon_loss = self.reconstr_loss()
		#self.reconstr_loss = self.reconstr_loss_slide()

		self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
				- tf.square(self.z_mean) 
				- tf.exp(self.z_log_sigma_sq), 1)
		self.cost = tf.reduce_mean(self.recon_loss + self.latent_loss)
#		self.lr = tf.Variable(self.config.learning_rate, False)
		self.optimizer = \
				tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)
	
	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.

		Return cost of mini-batch.
		"""

		#X = np.expand_dims(X, 1)
		#print np.sum(self.sess.run(self.z, feed_dict={self.x: X}))
                #print "--"
                '''
		print np.min(self.sess.run(self.x_recon_theta, feed_dict={self.x: X}))
		print np.sum(self.sess.run(self.x_recon_theta, feed_dict={self.x: X}))
		print np.max(self.sess.run(self.x_recon_theta, feed_dict={self.x: X}))
		print np.sum(self.sess.run(self.recon_loss, feed_dict={self.x: X}))
                '''
                '''
		print [np.sum(outp) for outp in self.sess.run(\
                        [self.recon_loss, self.x_recon_theta, tf.log(1e-10 + self.x_recon_theta), tf.log(1e-10 +1.0 - self.x_recon_theta)]\
                    , feed_dict={self.x: X})]
                '''
		#print np.sum(self.sess.run(self.latent_loss, feed_dict={self.x: X}))
		opt, cost, recon_loss = self.sess.run((self.optimizer, self.cost, tf.reduce_mean(self.recon_loss)), 
								  feed_dict={self.x: X,
                                                                            self.eps: np.random.normal(size=[X.shape[0],self.config.z_dim])})
		return cost, recon_loss

	def transform(self, X):
		"""Transform data by mapping it into the latent space."""
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.sess.run(self.z_mean, feed_dict={self.x: X,\
                                    self.eps: np.random.normal(size=[X.shape[0],self.config.z_dim])})


	def generate(self, z_mu=None):
		""" Generate data by sampling from latent space.

		If z_mu is not None, data for this point in latent space is
		generated. Otherwise, z_mu is drawn from prior in latent 
		space.        
		"""
		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture["n_z"])
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.sess.run(self.x_recon_theta, 
							 feed_dict={self.z: z_mu})

	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.sess.run(self.x_recon_theta, 
				feed_dict={self.x: X})

	def save(self, rep_dir):
		saver = tf.train.Saver()
		saver.save(self.sess, rep_dir+'/training.ckpt')

	def load(self, rep_dir):
		saver = tf.train.Saver()
		saver.restore(self.sess, rep_dir + '/training.ckpt')

def train(nga, rd):
	# Training cycle
        batch_size = 1024
	display_step = 5
        data = np.concatenate((rd.ancestry_mask, rd.ancestry_mask.T), axis=1)

	for epoch in range(200):
                np.random.shuffle(data)
		total_cost = 0.
		total_recon_loss = 0.
		count = 0
		#batch_size = nga.config.batch_size
		# Loop over all batches

#		lr_new = nga.config.learning_rate * (nga.config.lr_decay ** int(epoch/100))
#		nga.sess.run(tf.assign(nga.lr, lr_new))

		while True:
			#batch_xs, _ = gen.read_batch(batch_size) #mnist.train.next_batch(batch_size)
                        last = min((count+1)*batch_size, data.shape[0])
                        batch = data[count*batch_size:last]
			# Fit training using batch data
			cost, recon_loss = nga.partial_fit(batch)

			total_cost += cost 
			total_recon_loss += recon_loss 
			count += 1
                        if last >= data.shape[0]:
                            break

		# Display logs per epoch step

		if  epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch), \
				"cost=", total_cost/count, \
				"recon loss=", total_recon_loss/count
			sys.stdout.flush()

#	nga.save('checkpoints')

def create_sample_for_plot(nga, rd):
        #x_sample = rd.ancestry_mask
        x_sample = np.concatenate((rd.ancestry_mask, rd.ancestry_mask.T), axis=1)
        print x_sample
        y_sample = np.sum(x_sample, axis =1)
        y_sample /= np.max(y_sample)

	z_mu = nga.transform(x_sample)

	h5f = h5py.File('plot_data.h5', 'w')
	h5f.create_dataset('z', data=z_mu)
	h5f.create_dataset('y', data=y_sample)
	h5f.close()

def main():
	oboFile = open("data/hp.obo")
	rd = reader.Reader(oboFile, False) #, vectorFile)
        config = phraseConfig.Config
        config.update_with_reader(rd)
	nga = NGA(Config)
	train(nga, rd)
	create_sample_for_plot(nga, rd)


if __name__ == '__main__':
	main()
