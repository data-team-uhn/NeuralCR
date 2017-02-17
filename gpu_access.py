def get_gpu(pref):
	import gpu_lock
	import sys
	board = str(gpu_lock.obtain_lock_id(pref=pref))

	if board == "-1":
		sys.stderr.write("No GPUs available!\n")
		exit()
	sys.stderr.write("Using GPU:"+board+"\n")
	return board
