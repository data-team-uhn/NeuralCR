def get_gpu(pid, pref):
	import gpu_lock
	import sys
	board = str(gpu_lock.obtain_lock_id(pid, pref))

	if int(board) != int(pref):
            sys.stdout.write("GPU:"+str(pref)+" is locked!\n")
                #sys.stderr.write("No GPUs available!\n")
            exit()
	sys.stdout.write("Using GPU:"+board+"\n")
	return board
