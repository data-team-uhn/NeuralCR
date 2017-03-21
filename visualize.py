import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation
import numpy as np

def plot(vis_data, y):
	

	vis_x = vis_data[:, 0]
	vis_y = vis_data[:, 1]
	#y = y[:10]

	#plt.rc('text', usetex=True)
	#plt.rc('font', family='serif')

#	fig = plt.figure()
#	fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

#	ax = fig.add_subplot(111)


	plt.scatter(vis_x, vis_y, c=y) #, cmap=plt.cm.get_cmap("jet", 10))
	#ax.plot(vis_x, vis_y, lw=0.01) #, c=y)
	#plt.xlabel(r'$\displaystyle l_0$', fontsize=12)
	#plt.ylabel(r'$l_1$', fontsize=12)
	plt.colorbar() #ticks=range(10))
	plt.xlim(min(vis_x)-2, max(vis_x)+2)
	plt.ylim(min(vis_y)-2, max(vis_y)+2)
	#plt.clim(0.0, 1.0)
	plt.show()


def animate_plot(vis_data):
	fig = plt.figure()

	x = vis_data[:, 0]
	y = vis_data[:, 1]
	plt.xlim(min(x)-2, max(x)+2)
	plt.ylim(min(y)-2, max(y)+2)
	'''
	plt.xlim(-2, +2)
	plt.ylim(-2, +2)
	'''
#	plt.ylim(0, 1)

	graph, = plt.plot([], [], 'o')

	def animate(i):
		graph.set_data(x[:i+1], y[:i+1])
		return graph

	ani = FuncAnimation(fig, animate, frames=10000, interval=10)
	plt.show()

def main():
	#h5f = h5py.File('plot_data_rnn.h5', 'r')
	h5f = h5py.File('plot_data.h5', 'r')
	z = np.array(h5f['z'])
	#z = np.squeeze(z)
	y = np.array(h5f['y'])
	#y = 1.0*np.array(range(z.shape[0]))/z.shape[0]
	h5f.close()

#	z = z[np.argsort(y),:]
#	y = y[np.argsort(y)]


	from tsne import bh_sne
	vis_data = bh_sne(np.float64(z))
	#vis_data = z #bh_sne(np.float64(z))

#	animate_plot(vis_data)
#	plot(z, y)
	plot(vis_data, y)

if __name__ == '__main__':
	main()
