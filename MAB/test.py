


# 处理数据代码
import json
from collections import defaultdict

def process():
    data_list = json.load(open('/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json'))

    sample_distribution_dict = defaultdict(int)

    for id,data in enumerate(data_list):
        sample_distribution_dict[data['answer']] += 1
    

    with open('./train_sample_distribution.json','w') as f:
        json.dump(sample_distribution_dict,f,indent=4,ensure_ascii=False)
        f.close()

process()