def get_gpu():
	import gpu_lock
	import sys
	board = str(gpu_lock.obtain_lock_id())

	if board == "-1":
		sys.stderr.write("No GPUs available!\n")
		exit()
	sys.stderr.write("Using GPU:"+board+"\n")
	return board
