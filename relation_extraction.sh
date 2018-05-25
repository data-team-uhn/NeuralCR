###### Config ######
fasttext=../data/model_pmc.bin
model_params=../exp_mimic4/model_params/
input=/hpf/projects/brudno/arbabi/mimic/NOTEEVENTS.csv
experiment_dir=$1
sroot=386053000

mkdir -p $experiment_dir

python annotate_text.py --fasttext $fasttext --params $model_params --input $input --output $experiment_dir/anotations.csv 
python relation_extraction.py --input $experiment_dir/anotations.csv --params $model_params --sroot $sroot --output $experiment_dir/relations.p
python relation_vis.py $experiment_dir/relations.p $model_params --top 200 > $experiment_dir/top_relations.txt
