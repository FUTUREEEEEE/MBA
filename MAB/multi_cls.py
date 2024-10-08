from datasets import Dataset
import os 
import ast
import argparse
import time
import json
from transformers import AutoTokenizer,TrainerCallback
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from dotted_dict import DottedDict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['WANDB_DISABLED'] = 'true'
def sigmod(x):
    return 1/(1+np.exp(-x))

def compute_metrics_(prediction):
    """
    Compute accuracy, precision, recall, and F1-score for multi-label classification.
    

    
    Returns:
    dict: A dictionary with accuracy, precision, recall, and f1-score.
    """
    
    preds_ = prediction.predictions # for decoder output
    # preds_ = prediction.predictions # for encoder output
    print(preds_.shape)
    print(preds_[0].shape,preds_[1].shape)
    true_labels_np = prediction.label_ids
    #  each row of preds_>0.5 to 1 others to 0
    pred_labels_np = np.zeros_like(preds_)
    pred_labels_np[np.where(preds_ >= 0.5)] = 1

    print(f"true_labels_np:\n{true_labels_np}\n,pred_labels_np:\n{pred_labels_np}")
    
    # Calculate Accuracy: This is less informative as it considers all labels simultaneously
    accuracy = accuracy_score(true_labels_np, pred_labels_np)
    
    # Calculate Precision, Recall, and F1-Score: Using 'macro' and 'micro' averaging
    precision_micro = precision_score(true_labels_np, pred_labels_np, average='micro')
    recall_micro = recall_score(true_labels_np, pred_labels_np, average='micro')
    f1_micro = f1_score(true_labels_np, pred_labels_np, average='micro')
    
    precision_macro = precision_score(true_labels_np, pred_labels_np, average='macro')
    recall_macro = recall_score(true_labels_np, pred_labels_np, average='macro')
    f1_macro = f1_score(true_labels_np, pred_labels_np, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1': f1_macro

    }

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = 0#roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

def create_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--exp_name', type=str, default='cls', help='Experiment name')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--use_cwq', choices=['only', 'both', 'none'], default='none', help='Use ComplexWebQuestions dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased', help='Model path')
    parser.add_argument('--dataset',type=str,default='webqsp')
    parser.add_argument('--llm_prefix', type=str, default=None, help='The prefix for the LLM')
    parser.add_argument('--sample_dataset',type=float,default=None)
    parser.add_argument('--train_method_index', type=str, default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}", help='String representation of the dictionary')
    parser.add_argument('--test_method_index', type=str, default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}", help='String representation of the dictionary')
    
    return parser

def load_model(labels, id2label, label2id, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                        problem_type="multi_label_classification",
                                                        num_labels=len(labels),
                                                        id2label=id2label,
                                                        label2id=label2id)

    return tokenizer,model

labels = list(["zero","one","multiple"])

id2label = {}
label2id = {}



for index,i in enumerate(labels):
    id2label[index] = i
    label2id[i] = index
    
def main_loop(seed,user_args):



    # train_set,_ = load_dataset(dataset=user_args.dataset,train=True)
    # train_set = train.json
    with open('train.json') as f:
        train_set = json.load(f)
    with open('valid.json') as f:
        valid_set = json.load(f)
    with open('predict.json') as f:
        test_set = json.load(f)

    # shuffle the train_set
    np.random.seed(seed)
    np.random.shuffle(train_set)

    train_set = Dataset.from_list(train_set,split='train')
    valid_set = Dataset.from_list(valid_set,split='valid')
    test_set = Dataset.from_list(valid_set,split='test')

    # if user_args.use_cwq == 'only':
    #     test_set,_ = load_dataset(train=False,dataset='cwq')
    #     test_set = Dataset.from_list(test_set,split='test')
    # elif user_args.use_cwq == 'both':
    #     test_set,_ = load_dataset(dataset=user_args.dataset,train=False)
    #     cwq_test_set,_ = load_dataset(train=False,dataset='cwq')
    #     test_set = test_set + cwq_test_set
    #     test_set = Dataset.from_list(test_set,split='test')
    # else:
    #     test_set, _ = load_dataset(dataset=user_args.dataset,train=False)
    #     test_set = Dataset.from_list(test_set,split='test')

    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, num_steps=10):
            self.num_steps = num_steps
        
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.num_steps:
                return DottedDict({"should_training_stop": True,"should_log":False,"should_evaluate":False,"should_save":True,"should_epoch_stop":True})
            else:
                return DottedDict({"should_log":False,"should_evaluate":False,"should_save":False,"should_epoch_stop":False,"should_training_stop":False})

    
    print(id2label)
    print(label2id)

    tokenizer, model = load_model(labels, id2label, label2id, user_args.model_path)


    def preprocess_data(examples):
    # take a batch of texts
        text = examples["question"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        
        # fill numpy array
        for index,i in enumerate(labels_matrix):
            for key in labels:
                if key in examples['total_answer'][index]:
                    labels_matrix[index][label2id[key]] = 1
                    

        
        
        encoding["labels"] = labels_matrix.tolist()
        encoding["label_ids"] = labels_matrix.tolist()



        return encoding



    # import pdb;pdb.set_trace()
    train_set = train_set.map(preprocess_data, batched=True, remove_columns=train_set.column_names)
    val_set = valid_set.map(preprocess_data, batched=True, remove_columns=valid_set.column_names)
    test_set = test_set.map(preprocess_data, batched=True, remove_columns=test_set.column_names)

    train_set.set_format("torch")
    val_set.set_format("torch")
    test_set.set_format("torch")
    

    metric_name = "f1"

    args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=user_args.lr,
    per_device_train_batch_size=user_args.batch_size,
    per_device_eval_batch_size=user_args.batch_size,
    num_train_epochs=user_args.epochs,
    weight_decay=0.01,
    metric_for_best_model=metric_name,
    load_best_model_at_end=True,
    report_to=None,
    dataloader_drop_last=False,
    push_to_hub=False,
    seed=seed,
    data_seed=seed,
    # logging_steps=5,
    # logging_dir='./logs',
    # eval_steps=10,
)

    trainer = Trainer(
    model,
    args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_,
    # callbacks=[EarlyStoppingCallback()],
)

    trainer.train()

    results = trainer.predict(test_set)


    # num_hit = 0


    # actions = [0 for i in range(len(labels))]
    # for index,i in enumerate(results.predictions):
    #     action_1 = np.argsort(i)[::-1][0]
    #     if test_set['labels'][index][action_1] == 1:
    #         num_hit += 1
    #     actions[action_1] += 1
    return test_set,results


parser = create_parser()
user_args = parser.parse_args()

avg_hit = []
avg_recall = []
avg_delay = []

test_set,results = main_loop(seed=0,user_args=user_args)


with open('predict.json') as f:
    testset = json.load(f)

id2label={0:'A',1:'B',2:'C'}
id_dict = {}
for index,i in enumerate(results.predictions):
    # use i as distribution and sample an action

    action_1 = np.argsort(i)[::-1][0]
    id_dict[testset[index]['id']] = testset[index]
    id_dict[testset[index]['id']]['prediction'] = id2label[action_1]

with open('dict_id_pred_results.json', 'w') as f:
    json.dump(id_dict, f, indent=4)

