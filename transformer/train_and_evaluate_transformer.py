import torch
import random
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from typing import Tuple
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
sns.set_theme()


def match_input() -> str:
    user_input = input()
    try:
        user_input = int(user_input)
    except ValueError:
        print("Please enter a number corresponding to the model you want to train")
        return ""
    match user_input:
        case 1: model_name = "BERT"
        case 2: model_name = "DeBERTa"
        case 3: model_name = "RoBERTa"
        case 4: model_name = "MPNet"
        case _: 
            print("Please input a number from 1-4")
            model_name = ""
    return model_name

def f1_one_vs_all(true, pred, class_label):
    """
    Calculate F1 score for one individual class
    """
    # Create "true" list with 1 for the labels we care about and 0 for the others
    true = [0 if v != class_label else 1 for v in true]
    # Create "pred" list with 1 for instances predicted to be the label we care about and 0 for the others
    pred = [0 if v != class_label else 1 for v in pred]
    # Calculate the f1 for these lists to get the f1 score for only the class we're interested in
    return f1_score(true, pred)

def flat_f1_score(y_true, y_pred, average='macro'):
    y_pred = np.argmax(y_pred, axis=1).flatten()
    return f1_score(y_true, y_pred, average=average)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def tokenize(input_text: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = []
    attention_masks = []

    for doc in tqdm(input_text):
        encoded_dict = tokenizer.encode_plus(
                            doc,                       # Document to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attention masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

def evaluation(test_dataloader, device, model_name, model, criterion, logsoftmax):
    print("Evaluating...")
    test_set_predictions, test_set_proba = [], []
    for batch in tqdm(test_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                if model_name == "MPNet":
                    output = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                else:
                    output = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss, logits = output.loss, output.logits
            loss = criterion(logsoftmax(logits), b_labels)
            pred_proba = softmax(logits, dim=-1).detach().cpu().numpy()  # pred class probabilities
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()   # pred class labels
            test_set_predictions.append(predictions)
            test_set_proba.append(pred_proba)

            del batch
   
    test_set_predictions = np.concatenate(test_set_predictions, axis=0)
    test_set_proba = np.concatenate(test_set_proba, axis=0)
    return test_set_predictions, test_set_proba

def cross_balancing(df, device, model_name, model, criterion, logsoftmax, dev_or_test='test'):
    """
    evaluation on the whole test set in chunks that contain the number of instances for each class

    step: the number of intances for each class per iteration, equal to the smallest class
    
    The method iterates over the instances in the three classes until all instances of the biggest class were seen.
    \nThis way, every instance in the test set is part of the evaluation.
    \nAn evaluation is done on each chunk, the resulting values are averaged to achieve results that are representative 
    of the models performance on the whole test set without any bias caused by class imbalance.
    """
    if dev_or_test == 'test':
        step = 101
    else:
        dev_or_test = 'dev'
        step = 102
    full_test_y_true = []
    full_test_y_pred = []
    all_macroF1 = 0
    all_roc_auc = 0
    all_label1_F1 = 0
    all_label2_F2 = 0
    all_label3_F3 = 0
    all_accuracy = 0
    iterations = 0
    for i in range(1, len(df[df['set'] == dev_or_test].loc[(df['label'] == 1)]['label'].to_list())+1, step):
        # get df slices containing 101 entries for each label
        new_df = df[df['set'] == dev_or_test].loc[(df['label'] == 0)][i:i+step]
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 1)][i:i+step]]) 
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 2)][i:i+step]]) 
        for i in [0, 1, 2]:
            if len(new_df.loc[(new_df['label'] == i)]['label'].to_list()) < step:
                wrap_around = step - len(new_df.loc[(new_df['label'] == i)]['label'].to_list())
                new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == i)][:wrap_around]])

        # evaluation
        test_docs = new_df['sent_concat'].to_list()
        test_input_ids, test_attention_masks = tokenize(test_docs)

        test_labels = torch.tensor(new_df['label'].to_list())
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        test_dataloader = DataLoader(
                    test_dataset,
                    sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially
                    batch_size=batch_size
                )
        
        test_set_predictions, test_set_proba = evaluation(test_dataloader, device, model_name, model, criterion, logsoftmax)
        test_labels = new_df['label'].to_list()

        full_test_y_true.extend(test_labels)
        full_test_y_pred.extend(test_set_predictions)

        macro_f1 = f1_score(test_labels, test_set_predictions, average='macro')
        all_macroF1 += macro_f1
        roc_auc_test = roc_auc_score(test_labels, test_set_proba, average='macro', multi_class='ovo')
        all_roc_auc += roc_auc_test

        f1_less, f1_eq, f1_more = f1_one_vs_all(test_labels, test_set_predictions, class_label=0), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=1), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=2)
        all_label1_F1 += f1_less
        all_label2_F2 += f1_eq
        all_label3_F3 += f1_more

        accuracy = accuracy_score(test_labels, test_set_predictions)
        all_accuracy += accuracy

        iterations += 1
    print()
    print("average stats")
    avr_MacroF1 = all_macroF1 / iterations
    avr_roc_auc = all_roc_auc / iterations
    print("macro F1:")
    print('{} set: {:.3}'.format(dev_or_test, avr_MacroF1))
    print("auc-roc-score:")
    print('{} set: {:.3}'.format(dev_or_test, avr_roc_auc))
    avr_accuracy = all_accuracy / iterations
    print(f'Accuracy: {avr_accuracy:.3f}')
    avr_label1_F1 = all_label1_F1 / iterations
    avr_label2_F2 = all_label2_F2 / iterations
    avr_label3_F3 = all_label3_F3 / iterations
    print(f'\nclass-wise F1 scores for the {dev_or_test} set:')
    print(f'1: {avr_label1_F1:.2f}\n2: {avr_label2_F2:.2f}\n3: {avr_label3_F3:.2f}')

    # create confusion matrix
    norm_setting = 'true'
    test_conf_matr = confusion_matrix(full_test_y_true, full_test_y_pred, normalize=norm_setting)
    test_conf_matr = pd.DataFrame(test_conf_matr, columns=['Less likely', 'Equally likely', 'More likely'],
                                index=['Less likely', 'Equally likely', 'More likely'])
    print("")
    print("True\\Predicted:")
    print(test_conf_matr)


if __name__ == '__main__':
    balanced = True  # switch between balanced and full dataset
    run_ID = 24  # to identify the results later
    epochs = 3
    balance_test = False  # enable balancing the test and dev datasets
    cross_balance = True  # enable crossvalidation on the test set, balance_test has to be False
    model_path = ""  # give a path to load a saved model

    # take console input to pick a model
    print("This program is used to train and evaluate transformer models from huggingface")
    print("The models that performed best so far and that we plan to use are DeBERTa and MPNet")
    print("")
    user_input = ""
    while user_input == "":
        print("Plase choose which model to train by typing its number:")
        print("1: BERT, 2: DeBERTa, 3: RoBERTa, 4: MPNet")
        user_input = match_input()
    model_name = user_input
    
    print(f'Starting training and evaluation for {model_name}')

    # Set random seed values for reproducibility
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # intoduce gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # read the data
    print("Preparing the dataset...")
    train = pd.read_json('../datasets/adept/train-dev-test-split/train.json')
    dev = pd.read_json('../datasets/adept/train-dev-test-split/val.json')
    test = pd.read_json('../datasets/adept/train-dev-test-split/test.json')

    train['set'] = 'train'
    dev['set'] = 'dev'
    test['set'] = 'test'
    df = pd.concat([train, dev, test])

    # Only keep label classes 1, 2 and 3 (comparison labels). Map them to 0, 1 and 2 for training convenience
    df = df[df['label'].isin([1, 2, 3])]
    df['label'] -= 1
    # filter duplicates
    df = df.drop_duplicates(subset=['sentence2'], keep='first') 
    # Balance the training data by randomly sampling 1500 examples from class 1
    train = df[df['set'] == 'train']
    dev = df[df['set'] == 'dev']
    test = df[df['set'] == 'test']
    if balanced:
        train = pd.concat([
            train[train['label'].isin([0, 2])],
            train[train['label'] == 1].sample(1500, random_state=seed_val)
        ])
    train['label'].value_counts()
    # Balance the dev and test by randomly sampling datapoints to match the smallest class
    if balance_test:
        test = pd.concat([
            test[test['label'].isin([2])],
            test[test['label'] == 0].sample(101, random_state=seed_val),
            test[test['label'] == 1].sample(101, random_state=seed_val)
        ])
        dev = pd.concat([
            dev[dev['label'].isin([2])],
            dev[dev['label'] == 0].sample(102, random_state=seed_val),
            dev[dev['label'] == 1].sample(102, random_state=seed_val)
        ])
    df = pd.concat([train, dev, test])
    df.reset_index(inplace=True, drop=True)

    # Tokenization
    df['sent_concat'] = df.apply(lambda row: row['sentence1'] + ' ' + row['sentence2'], axis=1)
    # load the tokenizer for the given model
    match model_name:
        case "BERT":
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        case "DeBERTa":
            from transformers import DebertaTokenizer
            tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        case "RoBERTa":
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        case "MPNet":
            from transformers import MPNetTokenizer
            tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
        case _:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # apply additional tokenization function to match our data to the desired shape
    # [CLS] sentence1 [SEP] sentence2 [SEP]
    train_docs = df[df['set'] == 'train']['sent_concat'].to_list()
    train_input_ids, train_attention_masks = tokenize(train_docs)
    dev_docs = df[df['set'] == 'dev']['sent_concat'].to_list()
    dev_input_ids, dev_attention_masks = tokenize(dev_docs)
    test_docs = df[df['set'] == 'test']['sent_concat'].to_list()
    test_input_ids, test_attention_masks = tokenize(test_docs)

    # create a torch dataloader for each split of the dataset
    batch_size = 16
    train_labels = torch.tensor(df[df['set'] == 'train']['label'].to_list())
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_dataloader = DataLoader(
                train_dataset,
                sampler=RandomSampler(train_dataset),  # Select batches randomly
                batch_size=batch_size
            )
    dev_labels = torch.tensor(df[df['set'] == 'dev']['label'].to_list())
    validation_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
    validation_dataloader = DataLoader(
                validation_dataset,
                sampler=SequentialSampler(validation_dataset),  # Pull out batches sequentially
                batch_size=batch_size
            )
    test_labels = torch.tensor(df[df['set'] == 'test']['label'].to_list())
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(
                test_dataset,
                sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially
                batch_size=batch_size
            )

    # load the given model
    print("Loading model...")
    match model_name:
        case "BERT":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels = 3,
                output_attentions = False,
                output_hidden_states = False,
            )
        case "DeBERTa":
            from transformers import DebertaForSequenceClassification
            model = DebertaForSequenceClassification.from_pretrained(
                "microsoft/deberta-base",
                num_labels = 3,
                output_attentions = False,
                output_hidden_states = False,
            )
        case "RoBERTa":
            from transformers import RobertaForSequenceClassification
            model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels = 3,
                output_attentions = False,
                output_hidden_states = False,
            )
        case "MPNet":
            from transformers import MPNetForSequenceClassification
            model = MPNetForSequenceClassification.from_pretrained(
                "microsoft/mpnet-base",
                num_labels = 3,
                output_attentions = False,
                output_hidden_states = False,
            )
        case _:
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels = 3,
                output_attentions = False,
                output_hidden_states = False,
            
            )
    model.to(device)

    # set training parameters
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5,
                    eps = 1e-8  # default
                    )
    logsoftmax = torch.nn.LogSoftmax(dim=1)  #dim=1
    criterion = torch.nn.NLLLoss()
    epochs = epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    training_stats = []
    total_t0 = time.time()

    # load a saved model if a path is given, otherwise train the given model
    if model_path != "":
        model = torch.load(model_path)
    else:
        # train the model
        for epoch_i in range(0, epochs):
            # training
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                if model_name == "MPNet":
                    output = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                else:
                    output = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                loss, logits = output.loss, output.logits
                loss = criterion(logsoftmax(logits), b_labels)

                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                del batch

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # validation
            print("")
            print("Validation...")

            t0 = time.time()
            model.eval()
            total_eval_f1 = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    if model_name == "MPNet":
                        output = model(b_input_ids,
                                        attention_mask=b_input_mask,
                                    labels=b_labels)
                    else:
                        output = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss, logits = output.loss, output.logits
                loss = criterion(logsoftmax(logits), b_labels)
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                predictions = np.argmax(logits, axis=1).flatten()
                b_labels = b_labels.to('cpu').numpy()

                total_eval_f1 += f1_score(b_labels, predictions, average='macro')
                del batch

            # Report the final accuracy for this validation run.
            avg_val_f1 = total_eval_f1 / len(validation_dataloader)
            print("  Macro F1: {0:.2f}".format(avg_val_f1))

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Macro F1.': avg_val_f1,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # evaluate the model on the test data
    if cross_balance:
        cross_balancing(df, device, model_name, model, criterion, logsoftmax, dev_or_test = 'test')
    else:
        test_set_predictions, test_set_proba = evaluation(test_dataloader, device, model_name, model, criterion, logsoftmax)
        
        # create confusion matrix
        norm_setting = 'true'
        test_conf_matr = confusion_matrix(df[df['set'] == 'test']['label'].to_list(), test_set_predictions,
                                        normalize=norm_setting)
        test_conf_matr = pd.DataFrame(test_conf_matr, columns=['Less likely', 'Equally likely', 'More likely'],
                                    index=['Less likely', 'Equally likely', 'More likely'])
        print("")
        print("True\\Predicted:")
        print(test_conf_matr)

        # print stats
        test_labels = df[df['set'] == 'test']['label'].to_list()
        print("")
        print("macro F1:")
        print('test set: {:.3}'.format(f1_score(df[df['set'] == 'test']['label'].to_list(), test_set_predictions, average='macro')))
        print("\nweighted F1:")
        print('test set: {:.3}'.format(f1_score(df[df['set'] == 'test']['label'].to_list(), test_set_predictions, average='weighted')))
        
        print("\nroc-auc score: ")
        roc_auc_test = roc_auc_score(test_labels, test_set_proba, average='macro', multi_class='ovo')
        print(f'test set: {roc_auc_test:.2f}')

        f1_less, f1_eq, f1_more = f1_one_vs_all(test_labels, test_set_predictions, class_label=0), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=1), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=2)
        print("\nclass-wise F1 scores for the test set:")
        print(f'1: {f1_less:.2f}\n2: {f1_eq:.2f}\n3: {f1_more:.2f}')



