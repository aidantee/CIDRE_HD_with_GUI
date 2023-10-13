
## Pytorch Implementation for Document-level Relation Extraction with Graph State LSTM and Contextual Word Embedding

### Introduction

We address the problem of Document-Level N-ary Relation Extraction with a graph-based Long-short term memory network that utilizes a unified dependency-structure of a document, combined with state of the art pre-trained contextual embedding for the Biomedical Domain.

Our model was trained in a end-to-end manner and use whole information from all mention pairs in the document to make the final prediction
**Require python 3.8.10**

<img src="glstm.png" alt="drawing" width="500"/>

### Dataset

We use BioCreative5 CDR to train, develop and evaluate our model. The CDR5 dataset contains 1500 documents (500 for training, 500 for development, and 500 for testing) annotated in Pubtator format, all entities relation was labeled at abstract-level instead of mention-level. We train our model with the training set and utilize the dev set to find the best parameters, then we use both the training set and dev set to train our model, and finally, we evaluate our model on the test set.

### Training
Please intall all prerequisite packages via requirements file

```
    pip install -r requirements.txt
```

All configurations of our model was decribed in the config.json file. To train our model, you can run the following command (manually modify your seed)
```
    cd src
    python build_data.py
    python main.py --concat --seed 23534
```
### Start UI service
You must specified path of model checkpoint  
Default:
- CONFIG_PATH=./config.json
- PREDICT_THRESHOLD=0.7
```
    cd src
    export CONFIG_PATH=PATH_OF_CONFIG_FILE 
    export PREDICT_THRESHOLD=THRESHOLD
    export MODEL_CKPT_PATH=PATH_OF_MODEL_CHECKPOINT
    export USE_CPU=TRUE/FALSE
    gradio serve.py
```

[//]: # (### Result)

[//]: # ()
[//]: # (|       | Precision | Recall | F1 |)

[//]: # (| :----------- | ----------- | ----------- | ---------- |)

[//]: # (| Our Model      |  52.41      | 71.51 | 60.35 |)

[//]: # (| Our Model + NER   |   60.09      |     64.54 | 62.23 |)
