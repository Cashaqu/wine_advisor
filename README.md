# Simple Wine-Advisor

## Description

This is project for getting wine recomendation from https://www.kaggle.com/datasets/zynicide/wine-reviews dataset
The main idea is to use the model doc2vec to find similar descriptions for a query

## Getting Started

### Dependencies
requirements.txt:
```
gensim==4.3.2
pandas==2.1.0
prettytable==3.8.0
nltk==3.8.1
sumy==0.11.0
tqdm==4.66.1
spacy==3.6.1
```

### Installing

You can clone project to your local machine.

### Executing program

For building and training doc2vec model use:
```
python.exe .\main.py --alpha=0.002 --vec_size=100 --num_epochs=100
```
For testing model use:
```
python.exe .\predict.py
```
