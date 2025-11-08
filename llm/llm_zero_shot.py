import os
import pandas as pd
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load Mistral
checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Read data

mount = '..'

train = pd.read_json(f'{mount}/datasets/adept/train-dev-test-split/train.json')
dev = pd.read_json(f'{mount}/datasets/adept/train-dev-test-split/val.json')
test = pd.read_json(f'{mount}/datasets/adept/train-dev-test-split/test.json')

train['set'] = 'train'
dev['set'] = 'dev'
test['set'] = 'test'
df = pd.concat([train, dev, test])

"""Only keep label classes 1, 2 and 3 (comparison labels). Map them to 0, 1 and 2 for training convenience:

* 1 => 0
* 2 => 1
* 3 => 2
"""

df = df[df['label'].isin([1, 2, 3])]
df['label'] -= 1

"""Filter out duplicated data points"""

df = df.drop_duplicates(subset=['sentence2'], keep='first')

train = df[df['set'] == 'train']
dev = df[df['set'] == 'dev']
test = df[df['set'] == 'test']


#Run LLM

prompt_template = """Below are two sentences:
Sentence1: {}
Sentence2: {}
How likely is Sentence2 in relation to Sentence1? Respond with a number:
0 - Sentence2 is less likely than Sentence1
1 - Sentence2 is equally as likely as Sentence1
2 - Sentence2 is more likely than Sentence1
Respond only with the number. Do not include anything else in your response.
"""

def ask_mistral(s1, s2):
  prompt = prompt_template.format(s1, s2)

  messages = [
    {
        "role": "user",
        "content": prompt
    },
  ]

  model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

  generated_ids = model.generate(model_inputs,
                                 pad_token_id=tokenizer.eos_token_id,
                                 max_new_tokens=10,
                                 do_sample=True)
  output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  output = output[len(prompt):].lstrip().rstrip()

  del prompt
  del messages
  del model_inputs
  del generated_ids

  return output

from tqdm import tqdm
tqdm.pandas()

# test set
test['mistral_raw'] = test.progress_apply(
    lambda row: ask_mistral(row['sentence1'], row['sentence2']),
    axis=1
    )
test.to_csv(f'{mount}/test_results/mistral_test.csv')

# dev set
dev['mistral_raw'] = dev.progress_apply(
    lambda row: ask_mistral(row['sentence1'], row['sentence2']),
    axis=1
    )
dev.to_csv(f'{mount}/test_results/mistral_dev.csv')

"""# Post-processing"""

subset = 'test' # test or dev
results = pd.read_csv(f'{mount}/test_results/mistral_{subset}.csv', index_col=0)

results['mistral_pred'] = results['mistral_raw'].apply(
    lambda x: int(x[0])
)

"""# Evaluation

## Standard
"""

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def f1_one_vs_all(true, pred, class_label):
  true = [0 if v != class_label else 1 for v in true]
  pred = [0 if v != class_label else 1 for v in pred]
  return f1_score(true, pred)

predictions = results['mistral_pred'].to_list()
labels = results['label'].to_list()

macro_f1_test = f1_score(labels, predictions, average='macro')
print(f'{subset.capitalize()} set: macro F1 = {macro_f1_test:.3f}')

f1_less, f1_eq, f1_more = f1_one_vs_all(labels, predictions, class_label=0), \
                          f1_one_vs_all(labels, predictions, class_label=1), \
                          f1_one_vs_all(labels, predictions, class_label=2)
f1_less, f1_eq, f1_more
print(f"""{subset.capitalize()} set: one-vs-all F1
1: {f1_less:.3f}
2: {f1_eq:.3f}
3: {f1_more:.3f}""")

acc_test = accuracy_score(labels, predictions)
print(f'{subset.capitalize()} set: accuracy = {acc_test:.3f}')

"""## Cross-balanced"""

from tqdm import tqdm

def cross_balanced_eval(df, model_name, dev_or_test='test'):
    if dev_or_test == 'test':
        step = 101 # equal to the number of instances in the smallest class
    else:
        dev_or_test = 'dev'
        step = 102

    full_test_y_true = []
    full_test_y_pred = []
    all_macroF1 = 0
    all_label1_F1 = 0
    all_label2_F2 = 0
    all_label3_F3 = 0
    all_accuracy = 0
    iterations = 0

    for i in tqdm(range(
        1, len(df[df['set'] == dev_or_test].loc[(df['label'] == 1)]['label'].to_list())+1, step)):
        # get df slices containing 101 entries for each label
        new_df = df[df['set'] == dev_or_test].loc[(df['label'] == 0)][i:i+step]
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 1)][i:i+step]])
        new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == 2)][i:i+step]])
        for i in [0, 1, 2]:
            if len(new_df.loc[(new_df['label'] == i)]['label'].to_list()) < step:
                wrap_around = step - len(new_df.loc[(new_df['label'] == i)]['label'].to_list())
                new_df = pd.concat([new_df, df[df['set'] == dev_or_test].loc[(df['label'] == i)][:wrap_around]])

        # evaluation
        test_set_predictions = new_df['mistral_pred'].to_list()
        test_labels = new_df['label'].to_list()

        full_test_y_true.extend(test_labels)
        full_test_y_pred.extend(test_set_predictions)

        macro_f1 = f1_score(test_labels, test_set_predictions, average='macro')
        all_macroF1 += macro_f1

        f1_less, f1_eq, f1_more = f1_one_vs_all(test_labels, test_set_predictions, class_label=0), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=1), \
                                f1_one_vs_all(test_labels, test_set_predictions, class_label=2)
        all_label1_F1 += f1_less
        all_label2_F2 += f1_eq
        all_label3_F3 += f1_more

        accuracy = accuracy_score(test_labels, test_set_predictions)
        all_accuracy += accuracy

        iterations += 1

    print("")
    print(f"{model_name} - {dev_or_test} set: average stats")
    avr_MacroF1 = all_macroF1 / iterations
    print(f"macro F1: {avr_MacroF1:.3}")
    avr_accuracy = all_accuracy / iterations
    print(f'Accuracy: {avr_accuracy:.3f}')

    avr_label1_F1 = all_label1_F1 / iterations
    avr_label2_F2 = all_label2_F2 / iterations
    avr_label3_F3 = all_label3_F3 / iterations
    print(f'\nclass-wise F1 scores')
    print(f'1: {avr_label1_F1:.3f}\n2: {avr_label2_F2:.3f}\n3: {avr_label3_F3:.3f}')

    # create confusion matrix
    norm_setting = 'true'
    test_conf_matr = confusion_matrix(full_test_y_true, full_test_y_pred, normalize=norm_setting)
    test_conf_matr = pd.DataFrame(test_conf_matr, columns=['Less likely', 'Equally likely', 'More likely'],
                                index=['Less likely', 'Equally likely', 'More likely'])
    print("")
    print("True\\Predicted:")
    print(test_conf_matr)

# dev
cross_balanced_eval(results, 'Mistral-7B-Instruct-v0.3', dev_or_test=subset)

# test
cross_balanced_eval(results, 'Mistral-7B-Instruct-v0.3', dev_or_test=subset)