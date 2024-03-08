import torch
import random
import math
import logging
import datetime
import pandas as pd
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, RandomSampler
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['token_embeddings'] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


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


def cross_balancing(df, model, dev_or_test='test'):
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
    step = 101
    full_test_y_true = []
    full_test_y_pred = []
    all_macroF1 = 0
    all_label1_F1 = 0
    all_label2_F2 = 0
    all_label3_F3 = 0
    all_acc = 0
    iterations = 0
    for i in range(1, len(df[df['set'] == dev_or_test].loc[(df['label'] == 0.5)]['label'].to_list())+1, step):
        # get df slices containing 101 entries for each label
        new_df = df[df['set'] == dev_or_test].loc[(df['label'] == 0)][i:i+step]
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 0.5)][i:i+step]]) 
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 1)][i:i+step]]) 
        for i in [0, 0.5, 1]:
            if len(new_df.loc[(new_df['label'] == i)]['label'].to_list()) < step:
                wrap_around = step - len(new_df.loc[(new_df['label'] == i)]['label'].to_list())
                new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == i)][:wrap_around]])
        new_df = new_df.drop(columns=['sent_concat'])

        # evaluation
        test_sent1 = new_df['sentence1'].to_list()
        test_sent2 = new_df['sentence2'].to_list()
        test_labels = new_df['label'].to_list()

        #Compute embedding for both lists
        test_embeddings1 = model.encode(test_sent1, convert_to_tensor=True)
        test_embeddings2 = model.encode(test_sent2, convert_to_tensor=True)
        
        #Compute cosine-similarities
        test_cosine_scores = util.cos_sim(test_embeddings1, test_embeddings2)

        # map the continous cosine similarities to the three classes
        test_y_pred = []
        for i in range(len(test_sent1)):
            if test_cosine_scores[i][i] <= 0.33:
                test_y_pred.append(0)
            elif test_cosine_scores[i][i] >= 0.66:
                test_y_pred.append(2)
            else: test_y_pred.append(1)
        test_y_true = [x*2 for x in new_df['label'].to_list()]
        # save true and predicted lists to make one big confusion matrix later
        full_test_y_true.extend(test_y_true)
        full_test_y_pred.extend(test_y_pred)
        
        macro_f1 = f1_score(test_y_true, test_y_pred, average='macro')
        all_macroF1 += macro_f1
        
        f1_less, f1_eq, f1_more = f1_one_vs_all(test_y_true, test_y_pred, class_label=0), \
                                f1_one_vs_all(test_y_true, test_y_pred, class_label=1), \
                                f1_one_vs_all(test_y_true, test_y_pred, class_label=2)
        all_label1_F1 += f1_less
        all_label2_F2 += f1_eq
        all_label3_F3 += f1_more

        acc = accuracy_score(test_y_true, test_y_pred)
        all_acc += acc
        iterations += 1
    print()
    print("average stats")
    count = iterations
    print(f'ITERATIONS: {iterations}')
    print()
    avr_MacroF1 = all_macroF1 / count
    print("macro F1:")
    print('{} set: {:.3}'.format(dev_or_test, avr_MacroF1))
    avr_acc = all_acc / count
    print("Accuracy:")
    print('{} set: {:.3}'.format(dev_or_test, avr_acc))
    avr_label1_F1 = all_label1_F1 / count
    avr_label2_F2 = all_label2_F2 / count
    avr_label3_F3 = all_label3_F3 / count
    print(f"\nclass-wise F1 scores for the {dev_or_test} set:")
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
    # switch model here: currently only supports all-mpnet-base-v2
    model_name = "all-mpnet-base-v2"
    balanced = True  # switch between balanced and full dataset
    run_ID = 23  # to identify the results later
    epochs = 4
    balance_test = False  # enable balancing the test and dev datasets
    cross_balance = True  # enable cross-balancing on the test set, balance_test has to be False
    model_path = ""  # give a path to load a saved model
    lower_threshold = 0.33
    upper_threshold = 0.66

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
    
    # Since the sentence transformer only outputs a distance metric between embeddings,
    # we have to take an extra step to map the outputs to our classes.
    # To make that easier we have for now chosen to convert our classes to values
    # ranging from 0 to 1, as this is similar to the original output range of the model.
    # we map class 1 to 0, class 2 to 0.5 and class 3 to 1
    df['label'] = df['label'] * 0.5
    df['label'] -= 0.5

    # filter duplicates
    df = df.drop_duplicates(subset=['sentence2'], keep='first') 
    # Balance the training data by randomly sampling 1500 examples from class 1
    train = df[df['set'] == 'train']
    dev = df[df['set'] == 'dev']
    test = df[df['set'] == 'test']

    if balanced:
        train = pd.concat([
            train[train['label'].isin([0, 1])],
            train[train['label'] == 0.5].sample(1500, random_state=seed_val)
        ])
    train['label'].value_counts()
    
    # Balance the dev and test by randomly sampling datapoints to match the smallest class
    if balance_test:
        test = pd.concat([
            test[test['label'].isin([1])],
            test[test['label'] == 0].sample(101, random_state=seed_val),
            test[test['label'] == 0.5].sample(101, random_state=seed_val)
        ])
        dev = pd.concat([
            dev[dev['label'].isin([1])],
            dev[dev['label'] == 0].sample(102, random_state=seed_val),
            dev[dev['label'] == 0.5].sample(102, random_state=seed_val)
        ])
    df = pd.concat([train, dev, test])
    df.reset_index(inplace=True, drop=True)

    # Preparing the Input
    # the model input contains both sentences in a list, as well as the target label
    df['sent_concat'] = df.apply(lambda row: InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']), axis=1)

    # create a torch dataloader for each split of the dataset
    batch_size = 16
    train_docs = df[df['set'] == 'train']['sent_concat'].to_list()
    train_dataloader = DataLoader(
                train_docs,
                sampler=RandomSampler(train_docs),  # Select batches randomly
                batch_size=batch_size
            )
    
    # load the given model
    print("Loading model...")
    if model_path != "":
        model = SentenceTransformer(model_path)
    elif model_path == "base":
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model.to(device)

        # set training parameters
        optimizer = AdamW(model.parameters(),
                        lr = 2e-5,
                        eps = 1e-8  # default
                        )
        train_loss = losses.CosineSimilarityLoss(model=model)
        num_epochs = epochs
        train_batch_size = 16

        logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

        logging.info("Read STSbenchmark train dataset")
        logging.info("Read STSbenchmark dev dataset")

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                evaluation_steps=1000,
                warmup_steps=warmup_steps)
    # put the model in evaluation mode and move model parameters back to the cpu, in case they were on the gpu before
    model.eval()
    model.cpu()

    if cross_balance:
        cross_balancing(df, model)
    else:
        # evaluate on the test set
        test_sent1 = df[df['set'] == 'test']['sentence1'].to_list()
        test_sent2 = df[df['set'] == 'test']['sentence2'].to_list()
        test_labels = df[df['set'] == 'test']['label'].to_list()

        #Compute embedding for both lists
        test_embeddings1 = model.encode(test_sent1, convert_to_tensor=True)
        test_embeddings2 = model.encode(test_sent2, convert_to_tensor=True)
        
        #Compute cosine-similarities
        test_cosine_scores = util.cos_sim(test_embeddings1, test_embeddings2)

        # map the continous cosine similarities to the three classes
        test_y_pred = []
        for i in range(len(test_sent1)):
            if test_cosine_scores[i][i] <= lower_threshold:
                test_y_pred.append(0)
            elif test_cosine_scores[i][i] >= upper_threshold:
                test_y_pred.append(2)
            else: test_y_pred.append(1)
        test_y_true = [x*2 for x in df[df['set'] == 'test']['label'].to_list()]
        
        # create confusion matrix
        norm_setting = 'true'
        test_conf_matr = confusion_matrix(test_y_true, test_y_pred, normalize=norm_setting)
        test_conf_matr = pd.DataFrame(test_conf_matr, columns=['Less likely', 'Equally likely', 'More likely'],
                                    index=['Less likely', 'Equally likely', 'More likely'])
        print("")
        print("True\\Predicted:")
        print(test_conf_matr)

        # print stats
        print("")
        print("macro F1:")
        print('test set: {:.3}'.format(f1_score(test_y_true, test_y_pred, average='macro')))
        print("\nweighted F1:")
        print('test set: {:.3}'.format(f1_score(test_y_true, test_y_pred, average='weighted')))

        test_labels = df[df['set'] == 'test']['label'].to_list()
        f1_less, f1_eq, f1_more = f1_one_vs_all(test_y_true, test_y_pred, class_label=0), \
                                f1_one_vs_all(test_y_true, test_y_pred, class_label=1), \
                                f1_one_vs_all(test_y_true, test_y_pred, class_label=2)
        print("\nclass-wise F1 scores for the test set:")
        print(f'1: {f1_less:.2f}\n2: {f1_eq:.2f}\n3: {f1_more:.2f}')
