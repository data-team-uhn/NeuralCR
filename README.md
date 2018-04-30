## Introduction
NCR is a concept recognizer for annotating unstructured text with concepts from an ontology. In its core, NCR uses a deep neural network trained to classify input phrases with concepts in a given ontology, and is capable of generalizing to synonyms not explicitly available. concept recognizer for annotating unstructured text with concepts from an ontology.

## Requirements
* Python 3.5 or newer
* NumPy & SciPy
* Tensorflow 1.5 or newer
* fasttext (https://pypi.python.org/pypi/fasttext)

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
  --neg_file    location the negative corpus
  --flat        if this flag is passed training will ignore the taxonomy infomration provided in the ontology
  ```

Example:
```
$ python  train.py --obofile hp.obo --oboroot HP:0000118 --fasttext word_vectors.bin --neg_file wikipedia.txt --output trained_model/
```
## Using the trained model

### Using in a python script
After training is finished, the model can be loaded inside a python script as follows:
```
import ncrmodel 
model = ncrmodel.NCRModel.loadfromfile(trained_model_dir, word_vectors_file)
```

Where `word_vectors` is the addresss to the fasttext word vector file and `trained_model_dir` is the address to the output directory of the training.

Then `model` can be used for matching a string to the closest concept:
```
model.get_match(['retina cancer', 'kidney disease'], 5)
```
The first argument of the above function call is a list of phrases to be matched and the second argument is the number of top matches to be reported.

The model can be also used for concept recognition in a larger text:
```
model.annotate_text('The paitient was diagnosed with retina cancer', 0.5)
```
Where the first argument is the input text string and the second argument is the concept calling score threshold.

## Concept recongition
Concept recognition can be also performed using `annotate_text.py`. 
```
The following arguments are mandatory:
  --params      address to the directroy where the trained model is stored
  --fasttext    address to the fasttext word vector file
  --input       address to the directory where the input text files are located
  --output      adresss to the directory where the output files will be stored
  --threshold   the concept calling score threshold for concept recognition [0.5]
```

Example:
```
$ python annotate_text.py --params trained_model. --fasttext word_vectors.bin --input documents/ --output annotations/
```


