def get_gpu(pid, pref):
	import gpu_lock
	import sys
	board = str(gpu_lock.obtain_lock_id(pid, pref))

	if board == "-1":
		sys.stderr.write("No GPUs available!\n")
		exit()
	sys.stderr.write("Using GPU:"+board+"\n")
	return board
