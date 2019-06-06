## Introduction
NCR is a concept recognizer for annotating unstructured text with concepts from an ontology. In its core, NCR uses a deep neural network trained to classify input phrases with concepts in a given ontology, and is capable of generalizing to synonyms not explicitly available. concept recognizer for annotating unstructured text with concepts from an ontology.

## Requirements
* Python 3.5 or newer
* Tensorflow 1.13 or newer
* fastText Python binding (https://github.com/facebookresearch/fastText/tree/master/python)

## Installation
Install the latest version of TensorFlow (NCR was developed using the version 1.13). You can use pip for this:
```
$ pip3 install tensorflow-gpu
```

If you do not have access to GPUs, you can install the CPU version instead:
```
$ pip3 install tensorflow
```

Install fastText for python:
```
$ git clone https://github.com/facebookresearch/fastText.git
$ pip3 install fastText/
```

Install NCR by simply cloning this repository:
```
$ git clone https://github.com/ccmbioinfo/NeuralCR.git
```

To run NCR you need a trained NCR model. You can train the model on your own custom ontology as explained [here](#training). Alternatively, you can download a pre-trained NCR model from [here](https://ncr.ccm.sickkids.ca/params/ncr_hpo_params.tar.gz), which is pre-trained on [HPO](https://hpo.jax.org/app/) (release of 2019-06-03):
```
$ wget https://ncr.ccm.sickkids.ca/params/ncr_hpo_params.tar.gz
$ tar -xzvf ncr_hpo_params.tar.gz
```

To verify if the pre-trained NCR is working, you can use the interactive session (more details [here](#interactive-session)) as follows:
```
$ python3 NeuralCR/interactive.py --params model_params/ --fasttext model_params/pmc_model_new.bin
```


## Training
The following files are needed to start the training:
* The ontology file in `.obo` format.
* A file containing the word vectors prepared by the fasttext library
* [Optional] A corpus free of the ontology concepts to be used as a negative reference (to reduce concept recognition false positives)

The training can be performed using `train.py`.
```
The following arguments are mandatory:
  --obofile     location of the ontology .obo file
  --oboroot     the concept in the ontology to be used as root (only this concept and its descendants will be used)
  --fasttext    location of the fasttext word vector file
  --output      location of the directroy where the trained model will be stored
  
 The following arguments are optional:
  --neg_file    location of the negative corpus (text file)
  --epochs      Number of training epochs [80]
  --n_ensembles Number of ensembles [10]
  --flat        if this flag is passed training will ignore the taxonomy infomration provided in the ontology
  ```

Example:
```
$ python3  train.py --obofile hp.obo --oboroot HP:0000118 --fasttext word_vectors.bin --neg_file wikipedia.txt --output model_params/
```
## Using the trained model

### Using in a python script
After training is finished, the model can be loaded inside a python script as follows:
```
import ncrmodel 
model = ncrmodel.NCR.loadfromfile(param_dir, word_model_file)
```

Where `word_model_file` is the addresss to the fasttext word vector file and `param_dir` is the address to the output directory of the training.

Then `model` can be used for matching a list of strings to the most similar concepts:
```
model.get_match(['retina cancer', 'kidney disease'], 5)
```
The first argument of the above function call is a list of phrases to be matched and the second argument is the number of top matches to be reported.

The model can be also used for concept recognition in a larger text:
```
model.annotate_text('The paitient was diagnosed with retina cancer', 0.8)
```
Where the first argument is the input text string and the second argument is the concept calling score threshold.

### Concept recongition
Concept recognition can be also performed using `annotate_text.py`. 
```
The following arguments are mandatory:
  --params      address to the directroy where the trained model parameters are stored
  --fasttext    address to the fasttext word vector file
  --input       address to the directory where the input text files are located
  --output      adresss to the directory where the output files will be stored
  
The following arguments are optional:
  --threshold   the score threshold for concept recognition [0.8]
```

Example:
```
$ python3 annotate_text.py --params model_params --fasttext word_vectors.bin --input documents/ --output annotations/
```

### Interactive session
Concept recognition can be done in an interactive session through `interactive.py`. After the model is loaded, concept recognition will be performed on the standard input.
```
The following arguments are mandatory:
  --params      address to the directroy where the trained model parameters are stored
  --fasttext    address to the fasttext word vector file
  
The following arguments are optional:
  --threshold   the score threshold for concept recognition [0.8]
```

* Example:
Run the script:
```
$ python3 interactive.py --params model_params --fasttext word_vectors.bin
```
Querry:
```
The patient was diagnosed with kidney cancer.
```
Output:
```
31	44	HP:0009726	Renal neoplasm	0.98976994
```

You can also link concepts to an isolated phrase by starting your query with `>`:

Querry:
```
>kidney cancer
```
Output:
```
HP:0009726 Renal neoplasm 0.98976994
HP:0005584 Renal cell carcinoma 0.0063989228
HP:0030409 Renal transitional cell carcinoma 0.0014158536
HP:0010786 Urinary tract neoplasm 0.00049688865
HP:0000077 Abnormality of the kidney 0.0003460226
```
## Online Web App and API
A web app is available for NCR trained on HPO:

https://ncr.ccm.sickkids.ca/curr/


## References
Please cite NCR if you have used it in your work.

```
@article{arbabi2019identifying,
  title={Identifying Clinical Terms in Medical Text Using Ontology-Guided Machine Learning},
  author={Arbabi, Aryan and Adams, David R and Fidler, Sanja and Brudno, Michael},
  journal={JMIR medical informatics},
  volume={7},
  number={2},
  pages={e12596},
  year={2019},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```

