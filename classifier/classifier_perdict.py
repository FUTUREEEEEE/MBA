import logging
import torch
from torch import sigmoid
import numpy as np

from typing import Optional

import numpy as np
import pandas as pd
import json
import logging
from sklearn.preprocessing import label_binarize
from tqdm import tqdm




from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_jsonline(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in tqdm(f.readlines()):
            result.append(json.loads(l))
        return result

def calculate_metrics(predictions, labels, classes, detailed=False):
    labels = label_binarize(labels, classes=classes)
    predictions = label_binarize(predictions, classes=classes)

    if detailed:
        results = {}
        for i, class_label in enumerate(classes):
            precision = precision_score_custom(predictions[:, i], labels[:, i])
            recall = recall_score_custom(predictions[:, i], labels[:, i])
            f1 = f1_score_custom(predictions[:, i], labels[:, i])
            results[class_label] = {"precision": precision, "recall": recall, "f1": f1}
        return results
    else:
        precision = np.nanmean([precision_score_custom(pred, lab) for pred, lab in zip(predictions.T, labels.T)])
        recall = np.nanmean([recall_score_custom(pred, lab) for pred, lab in zip(predictions.T, labels.T)])
        f1 = np.nanmean([f1_score_custom(pred, lab) for pred, lab in zip(predictions.T, labels.T)])
        return {"precision": precision, "recall": recall, "f1": f1}

def precision_score_custom(predictions, labels):
    true_positives = np.logical_and(predictions, labels).sum()
    predicted_positives = predictions.sum()
    return true_positives / (predicted_positives + 1e-10)

def recall_score_custom(predictions, labels):
    true_positives = np.logical_and(predictions, labels).sum()
    actual_positives = labels.sum()
    return true_positives / (actual_positives + 1e-10)

def f1_score_custom(predictions, labels):
    precision = precision_score_custom(predictions, labels)
    recall = recall_score_custom(predictions, labels)
    return 2 * (precision * recall) / (precision + recall + 1e-10)

def accuracy_score_custom(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    print(f"accuracy:{accuracy}")
    return accuracy

class Classify_Llm(torch.nn.Module):
    def __init__(self,model_path,problem_type:str="single_label_classification",num_labels:int=3) -> None:
        super().__init__()

        self.problem_type = problem_type
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,device_map="auto",trust_remote_code=True,torch_dtype=torch.bfloat16,num_labels=num_labels)
        self.model.eval()

    def forward(self,input_ids: torch.LongTensor = None,attention_mask: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        return outputs
    def predict(self,input_ids: torch.LongTensor = None,attention_mask: Optional[torch.Tensor] = None):
        logits = self.forward(input_ids,attention_mask).logits
        if self.problem_type == "regression":
            predictions = logits.squeeze()
        elif self.problem_type == "single_label_classification":
            predictions = torch.argmax(logits, dim=-1)
        elif self.problem_type == "multi_label_classification":
            predictions = (sigmoid(logits) >= 0.5).int()
        print(f"predictions:{predictions}")
        return predictions
    

label_to_option = {
    0: 'A',
    1: 'B',
    2: 'C',
}
def test_predict(model_path:str,data_path:str,output_path:str,problem_type:str="single_label_classification",num_labels:int=3,batch_size:int=8,device:torch.device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,padding_side="left")
    data_input = json.load(open(data_path,"r",encoding="utf-8"))

    model = Classify_Llm(model_path,problem_type,num_labels)

    num_batch = len(data_input) // batch_size if len(data_input) % batch_size == 0 else len(data_input) // batch_size + 1

    result = []

    for i in tqdm(range(num_batch)):
        
        batch_data = data_input[i*batch_size:(i+1)*batch_size]

        batch_inputs = [f"{d['question']}" for d in batch_data]
        
        batch_ids = tokenizer(batch_inputs,return_tensors="pt",padding=True).to(device)


        predictions = model.predict(batch_ids.input_ids,batch_ids.attention_mask)

        batch_output = []
        for k in predictions:
            result.append(k.item())


        
        for j in range(i*batch_size,(i+1)*batch_size):
            if j<len(data_input):
                data_input[j]['prediction'] = label_to_option[result[j]]
                batch_output.append(result[j])
            else:
                break
        print(f"\n\nbatch inputs:\n{batch_inputs}\n\nbatch output:{batch_output}\n\n")

    # with open(output_path+"/_multiLabel-train_singleLabel_test_dict_id_pred_results.json", "w",encoding='utf-8') as f:
    #         for i in data_input:
    #             f.write(json.dumps(i, ensure_ascii=False) + "\n")
    #         f.close()
    
    predict_and_evaluate(data_input)

def predict_and_evaluate(data_list:str):
    df = pd.DataFrame(data_list)
    classes = [0, 1, 2]  # Update this based on actual classes

    labels = df['answer'].values
    predictions = df['prediction'].values
    
    metrics = calculate_metrics(predictions, labels, classes, detailed=True)
    accuracy = accuracy_score_custom(predictions, labels)
    print("accuracy:", accuracy)
    print("Metrics:", metrics)

    result = {
        "accuracy": accuracy,
        "metrics": metrics
    }
    with open("//apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/multilabel/_metrics.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        f.close()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model_path = "/apdcephfs_cq10/share_1567347/share_info/tensorgao/Qwen/output_classify/09-13-2024_11_t5-large_mab-classify___bs__maxlen_400_pad_right_lr_2e-6_format_mab"
    data_path = "/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/valid.json"
    test_predict(model_path=model_path,data_path=data_path,output_path="/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/multilabel",device=device)