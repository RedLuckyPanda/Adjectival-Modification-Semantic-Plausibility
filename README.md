# Modeling Semantic Plausibility (WS 2023/24) - Anna Golub and Beate Zywietz

## Code Structure
This repository is dedicated to modeling semantic plausibility with the use of large language models and the [ADEPT](https://aclanthology.org/2021.acl-long.553/) dataset.

The `dataset_analysis` folder contains the `adept_dataset_analysis` jupyter notebook. It provides a first look at the data using descriptive statistics (first submission).

The `sentence-transformer` folder contains the code for fine-tuning a pre-trained sentence transformer model based on [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on the ADEPT dataset and the code for the model's performance evaluation. There are two files: `notebook_sentence-transformer.ipynb` and `train_and_evaluate_sentence-transformer.py`. The first one is a Jupyter notebook. It's meant for familiarizing oneself with the code as it walks the user through it step by step and shows the intermediate outputs. The second file contains the same code but formatted for streamlining and easy execution. 

The `transformer` folder contains the code for fine-tuning pre-trained transformer models ([BERT](https://huggingface.co/docs/transformers/model_doc/bert), [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), [MPNet](https://huggingface.co/microsoft/mpnet-base)) on the ADEPT dataset with performance evaluation. Same as the `sentence-transformer` folder, the `transformer` folder contains a Jupyter notebook with comments and the same code as a Python executable file.

The `significance_testing` folder contains the Jupyter notebook and data for running the significance testing on model performance metrics.

## Quick Start
To run the code, please do the following:
1. Make sure your installation of Python is >**version 3.10**< or higher. If necessary, [update Python](https://www.python.org/downloads/).
2. Clone the repository to your system with `git clone`
3. Go to the repository directory: `cd SemanticPlausibility_23-24`
4. Install the project dependencies: `pip install -r requirements.txt`
5. Follow [the instructions](https://pytorch.org/get-started/locally/) to install `pytorch`.
6. To run one of the Jupyter notebooks, use this command: `jupyter notebook`. The Jupyter interface will open in a browser window, where you can navigate through the file system and choose a notebook to run.
7. To train and evaluate a **transformer** model, run: `python transformer/train_and_evaluate_transformer.py`
8. To train and evaluate the **sentence-transformer** model, run: `python sentence-transformer/train_and_evaluate_sentence-transformer.py`

## Project Results

Available here: https://docs.google.com/spreadsheets/d/1AT1-zhi3iSA1e6WamGoMFsS19wnBn9xpbXDEWsSPjgY/edit?usp=sharing

## References
The code is partially based on these tutorials:
1. [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) By Chris McCormick and Nick Ryan
2. [The documentation for semantic textual similarity on SBERT.net](https://www.sbert.net/docs/usage/semantic_textual_similarity.html), the source of the sentence-transformer.
