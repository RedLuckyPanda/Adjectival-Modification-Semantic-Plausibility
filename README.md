# Modeling Semantic Plausibility (WS 2023/24) - Anna Golub and Beate Zywietz

## Overview
This repository is dedicated to modeling semantic plausibility with the use of large language models and the [ADEPT](https://aclanthology.org/2021.acl-long.553/) dataset.

`adept_dataset_analysis.ipynb` - first look at the data using descriptive statistics

`transformer_finetuning.ipynb` - fine-tuning pre-trained transformer models ([BERT](https://huggingface.co/docs/transformers/model_doc/bert), [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) on the ADEPT dataset with performance evaluation

`sentence-transformer_finetuning.ipynb` - fine-tuning a pre-trained sentence transformer model on the ADEPT dataset with performance evaluation

## Quick Start
To run the code, please do the following:
1. Clone the repository to your system with `git clone`
2. Go to the repository directory `SemanticPlausibility_23-24`
3. Install the project dependencies using `pip install -r requirements.txt`
4. Run a notebook of your choice using this command: #TODO
