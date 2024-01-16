# Modeling Semantic Plausibility (WS 2023/24) - Anna Golub and Beate Zywietz

## Overview
This repository is dedicated to modeling semantic plausibility with the use of large language models and the [ADEPT](https://aclanthology.org/2021.acl-long.553/) dataset.

`adept_dataset_analysis.ipynb` - first look at the data using descriptive statistics

`sentence-transformer_finetuning.ipynb` - fine-tuning a pre-trained sentence transformer model based on [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on the ADEPT dataset with performance evaluation

`transformer_finetuning.ipynb` - fine-tuning pre-trained transformer models ([BERT](https://huggingface.co/docs/transformers/model_doc/bert), [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), [MPNet](https://huggingface.co/microsoft/mpnet-base)) on the ADEPT dataset with performance evaluation

## Quick Start
To run the code, please do the following:
1. Clone the repository to your system with `git clone`
2. Go to the repository directory: `cd SemanticPlausibility_23-24`
3. Install the project dependencies: `pip install -r requirements.txt`
4. To train and evaluate a transformer model run `python train_and_evaluate_transformer.py`
5. To train and evaluate the sentence-transformer model run `train_and_evaluate_sentence-transformer.py`

## Code Documentation
While the .py files contain comments to explain a code, the .ipynb files provide a more detailed step-by-step documentation.


Open the .ipynb files in a notebook editor of your chioce, or install Jupyter Notebook. 
To install Jupiter notebook run `pip install notebook`, then run `jupyter notebook` to open the application. 

