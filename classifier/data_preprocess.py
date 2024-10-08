""" 
data pre process
"""

import json
def process(data_list:list):
    result = []
    for data in data_list:
        inputs = data['question']
        labels = []

        if data['total_answer']==[]:
            print("error no answer")
        
        if "zero" in data['total_answer']:
            labels.append(0)
        if "one" in data['total_answer']:
            labels.append(1)
        if "multiple" in data['total_answer']:
            labels.append(2)
        result.append({"inputs":inputs,"labels":labels})

    with open("./train_multi_classifier.json","w",encoding='utf-8') as f:
        # for line in result:
        #     f.write(json.dumps(line,ensure_ascii=False)+"\n")
        json.dump(result,f,ensure_ascii=False,indent=4)


def test():

    label_to_option = {
    0: 'A',
    1: 'B',
    2: 'C',
    }
    with open("/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/multilabel/_multiLabel-train_singleLabel_test_dict_id_pred_results.json","r") as f:
        data_list =[json.loads(line) for line in f]

    dict = {}
    for id,data in enumerate(data_list):
        data['prediction'] = label_to_option[data['prediction']]
        dict[data['id']] = data
    
    with open("/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/multilabel/_multiLabel-train_singleLabel_test_dict_id_pred_results_1.json","w") as f:
        json.dump(dict,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    # with open("/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json","r") as f:
    #     data_list =json.load(f)

    # with open("/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/train_multi_classifier.json",'r') as f:
    #     data_list = [json.loads(line) for line in f]

    # print(len(data_list))
    # process(data_list)

    test()
