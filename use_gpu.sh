usage=`python ~/gpu_lock.py | grep 'Use:' | awk '{print substr($3,6,10)}' | head -n $(( $1+1 )) | tail -n 1`
total_usage=`python ~/gpu_lock.py | grep 'Use:' | awk '{total+=substr($3,6,10)} END {print total}'`
echo "total usage: "$total_usage"%"
if (( $usage > 0 ))
then
	echo 'GPU is being used!'
	echo $usage'%'
	exit
fi

echo -e 'import gpu_access\ngpu_access.get_gpu('$$','$1')\n' | python
CUDA_VISIBLE_DEVICES=$1 python $2 
