convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
}

experiment_dir=../exp4/
checkpoint_dir=$experiment_dir/model_params/
ontology=../data/hp.obo
ont_root='HP:0000118'
fasttext=../data/model_pmc.bin
negs=../data/wiki_text
phrase_data=../data/labeled_data
report_file=$experiment_dir/report.txt

rm -f $report_file
echo "Experiments report :: " >> $report_file
### train model on ontology
SECONDS=0
python train.py --obofile $ontology --oboroot $ont_root  --fasttext $fasttext --neg_file $negs --output $checkpoint_dir
echo "Training done. Time elapsed:: "$(convertsecs $SECONDS)
echo "Training done. Time elapsed:: "$(convertsecs $SECONDS) >> $report_file

### phrase level experiments
echo "Phrase-level experiments" >> $report_file
echo -ne "r@1\t" >> $report_file
python accuracy.py --fasttext $fasttext --params $checkpoint_dir --input $phrase_data --topk 1 | tail -n 1 | awk '{print $3}' >> $report_file
echo -ne "r@5\t" >> $report_file
python accuracy.py --fasttext $fasttext --params $checkpoint_dir --input $phrase_data --topk 5 | tail -n 1 | awk '{print $3}' >> $report_file

### use validation set to tune concept recognition threshold

val_input_dir=../../datasets/abstracts_val/
val_output_dir=$experiment_dir/validation/

mkdir -p $val_output_dir

rm -f $val_output_dir/report.txt 
echo "" >> $report_file
echo "Validation experiments" >> $report_file

SECONDS=0
for threshold in 0.65 0.7 0.75 0.8 0.85
do
	python annotate_text.py --fasttext $fasttext --params $checkpoint_dir --input $val_input_dir/text/ --output $val_output_dir/outputs_th$threshold/ --threshold $threshold 
	echo -n $threshold' && ' >> $val_output_dir/report.txt
	python eval.py $val_input_dir/labels/ $val_output_dir/outputs_th$threshold/ --obofile $ontology --oboroot $ont_root --output_column 2 >> $val_output_dir/report.txt
done

val_threshold=`sort $val_output_dir/report.txt -k 7,7 -r | head -n 1 | awk '{print $1}'`
cat $val_output_dir/report.txt >> $report_file
echo 'validation threshold: '$val_threshold >> $report_file
echo 'validation threshold: '$val_threshold
echo "Time elapsed for validation:: "$(convertsecs $SECONDS)
echo "Time elapsed for validation:: "$(convertsecs $SECONDS) >> $report_file

### test concept recognition
echo "" >> $report_file
for dataset in littleomim abstracts_test udp
do
	SECONDS=0
	echo 'Dataset: '$dataset >> $report_file
	echo 'Dataset: '$dataset
	cr_input_dir=../../datasets/$dataset/
	cr_output_dir=$experiment_dir/$dataset/
	rm -f $cr_output_dir/report.txt
	python annotate_text.py --fasttext $fasttext --params $checkpoint_dir --input $cr_input_dir/text/ --output $cr_output_dir/outputs/ --threshold $val_threshold
	python eval.py $cr_input_dir/labels/ $cr_output_dir/outputs/ --obofile $ontology --oboroot $ont_root --output_column 2 >> $report_file
	echo "Time elapsed:: "$(convertsecs $SECONDS)
	echo "Time elapsed:: "$(convertsecs $SECONDS) >> $report_file
done

