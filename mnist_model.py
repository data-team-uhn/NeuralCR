import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import phrase_model
import reader
import phraseConfig
import h5py

'''
def plot_latent(nga, mnist):
	x_sample, y_sample = mnist.test.next_batch(5000)
	z_mu = nga.transform(x_sample)
	plt.figure(figsize=(8, 6)) 
	plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
	plt.colorbar()
	plt.show()
'''


def plot_latent_tsne(data):
	from tsne import bh_sne


	vis_data = bh_sne(np.array(data, dtype=np.float))

	# plot the result
	vis_x = vis_data[:, 0]
	vis_y = vis_data[:, 1]

	plt.scatter(vis_x, vis_y)
	plt.clim(-0.5, 9.5)
	plt.show()


def main():
	plot_latent_tsne(np.array(h5py.File('hpo_embed.h5', 'r')['hpo_embed']))
	return
	print 'hello'
	repdir = 'checkpoints_backup/'
	datadir = 'data/'
	oboFile = open(datadir+"/hp.obo")
	vectorFile = open(datadir+"/vectors.txt")

	rd = reader.Reader(oboFile)
	config = phraseConfig.Config
	config.update_with_reader(rd)

	model = phrase_model.NCRModel(config)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver()
	saver.restore(sess, repdir + '/training.ckpt')

	h5f = h5py.File('hpo_embed.h5', 'w')
	h5f.create_dataset('hpo_embed', data=sess.run(model.HPO_embedding))
	h5f.close()

#	plot_latent_tsne(model.HPO_embedding)



if __name__ == '__main__':
	main()
