convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
}

###### Config ######
experiment_dir=$1
data_dir=$2
datasets_dir=$3
#fasttext=$data_dir/model_pmc.bin
#fasttext=$data_dir/crawl-300d-2M-subword.bin
#fasttext=$data_dir/BioWordVec_PubMed_MIMICIII_d200.bin
fasttext=$data_dir/pmc_model_new.bin
negs=$data_dir/wiki_text
phrase_data=$data_dir/labeled_data
val_input_dir=$datasets_dir/abstracts_val/
val_output_dir=$experiment_dir/validation/
baseline=$4/output/biolark_api/
ontology=$data_dir/hp.obo
#ontology=$data_dir/hp.obo
ont_root='HP:0000118'
####################

report_file=$experiment_dir/report.txt
checkpoint_dir=$experiment_dir/model_params/
checkpoint_dir_without_early_stopping=$experiment_dir/model_params_without_early_stopping/
additional_args="${@:5}"

mkdir -p $experiment_dir
rm -f $report_file
echo "Experiments report :: " >> $report_file
echo "Args used :: "$additional_args >> $report_file

####################
### train model on ontology
SECONDS=0
python3 -u train.py --obofile $ontology --oboroot $ont_root\
	--fasttext $fasttext --neg_file $negs --output $checkpoint_dir\
      	--output_without_early_stopping $checkpoint_dir_without_early_stopping\
	--sentence_val_input_dir $val_input_dir/text/\
       	--sentence_val_label_dir $val_input_dir/labels/\
	--phrase_val $phrase_data --verbose --epochs 80 --n_ensembles 10\
	$additional_args


echo "Training done. Time elapsed:: "$(convertsecs $SECONDS)
echo "Training done. Time elapsed:: "$(convertsecs $SECONDS) >> $report_file

### phrase level experiments
echo "Phrase-level experiments (with early stopping based on sent-validation -- DO NOT REPORT)" >> $report_file
echo -ne "r@1\t" >> $report_file
python3 -u accuracy.py --fasttext $fasttext --params $checkpoint_dir --input $phrase_data --topk 1 | tail -n 1 | awk '{print $3}' >> $report_file
tail -n 1 $report_file
echo -ne "r@5\t" >> $report_file
python3 -u accuracy.py --fasttext $fasttext --params $checkpoint_dir --input $phrase_data --topk 5 | tail -n 1 | awk '{print $3}' >> $report_file
tail -n 1 $report_file

echo "Phrase-level experiments (without early stopping -- REPORT)" >> $report_file
echo -ne "r@1\t" >> $report_file
python3 -u accuracy.py --fasttext $fasttext --params $checkpoint_dir_without_early_stopping --input $phrase_data --topk 1 | tail -n 1 | awk '{print $3}' >> $report_file
tail -n 1 $report_file
echo -ne "r@5\t" >> $report_file
python3 -u accuracy.py --fasttext $fasttext --params $checkpoint_dir_without_early_stopping --input $phrase_data --topk 5 | tail -n 1 | awk '{print $3}' >> $report_file
tail -n 1 $report_file

### use validation set to tune concept recognition threshold

mkdir -p $val_output_dir

rm -f $val_output_dir/report.txt 
echo "" >> $report_file
echo "Validation experiments" >> $report_file

SECONDS=0
for threshold in 0.65 0.7 0.75 0.8 0.85 0.9
do
	mkdir -p $val_output_dir/outputs_th$threshold/
	python3 -u annotate_text.py --fasttext $fasttext --params $checkpoint_dir --input $val_input_dir/text/ --output $val_output_dir/outputs_th$threshold/ --threshold $threshold 
	echo -ne $threshold'\t' >> $val_output_dir/report.txt
	python3 -u eval.py $val_input_dir/labels/ $val_output_dir/outputs_th$threshold/ --obofile $ontology --oboroot $ont_root --output_column 2 --no_error  >> $val_output_dir/report.txt
done

val_threshold=`sort -h $val_output_dir/report.txt -k 4,4 -r | head -n 1 | awk '{print $1}'`
cat $val_output_dir/report.txt >> $report_file
echo 'validation threshold: '$val_threshold >> $report_file
echo 'validation threshold: '$val_threshold
echo "Time elapsed for validation:: "$(convertsecs $SECONDS)
echo "Time elapsed for validation:: "$(convertsecs $SECONDS) >> $report_file

### test concept recognition
combined_output_dir=$experiment_dir/combined_udp_abstracts/outputs/
combined_input_labels=$datasets_dir/combined_udp_abstracts/labels/
mkdir -p $combined_output_dir
for dataset in abstracts_test udp
do
	echo "" >> $report_file
	echo "############################" >> $report_file
	SECONDS=0
	echo 'Dataset: '$dataset >> $report_file
	echo 'Dataset: '$dataset
	cr_input_dir=$datasets_dir/$dataset/
	cr_output_dir=$experiment_dir/$dataset/
	rm -f $cr_output_dir/report.txt
	mkdir -p $cr_output_dir/outputs/
	python3 -u annotate_text.py --fasttext $fasttext --params $checkpoint_dir --input $cr_input_dir/text/ --output $cr_output_dir/outputs/ --threshold $val_threshold
	python3 -u eval.py $cr_input_dir/labels/ $cr_output_dir/outputs/ --obofile $ontology --oboroot $ont_root --output_column 2 --no_error >> $report_file
	echo "" >> $report_file
	echo "Signficance Tests (Trained vs baseline) ::" >> $report_file
	echo "baseline="$baseline
	python3 -u t_test_eval.py $cr_input_dir/labels/ $cr_output_dir/outputs/ $baseline/$dataset/  --obofile $ontology --oboroot $ont_root --output_column_a 2 >> $report_file
	cp $cr_output_dir/outputs/* $combined_output_dir/
	echo "Time elapsed:: "$(convertsecs $SECONDS)
	echo "Time elapsed:: "$(convertsecs $SECONDS) >> $report_file
done

echo "" >> $report_file
echo "############################" >> $report_file
echo "Signficance Tests - combined dataset (Trained vs baseline) ::" >> $report_file
echo "baseline="$baseline
python3 -u t_test_eval.py $combined_input_labels $combined_output_dir $baseline/combined_udp_abstracts/  --obofile $ontology --oboroot $ont_root --output_column_a 2 >> $report_file


