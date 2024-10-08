import json
import jsonlines
import os 
from postprocess_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="model name.",default='flan_t5_xl', choices=("flan_t5_xl", "flan_t5_xxl", "gpt","MAB"))
parser.add_argument("--file_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--split", type=str,default="test")


args = parser.parse_args()

# your file path to /path_to_classifier/predict/dict_id_pred_results.json
classification_result_file = args.file_path
# stepNum_result_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}', 'total', 'stepNum.json')
stepNum_result_file = "/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/predictions/test/ircot_qa_flan_t5_xl/total/stepNum.json"
# output_path = os.path.join("predictions", 'classifier', '/'.join(classification_result_file.split('/')[classification_result_file.split('/').index('model')+1:-2]))
# output_path = os.path.join("predictions", 'classifier', 'ircot_qa', args.model_name)
output_path = args.output_path

nq_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__nq_to_nq__test_subsampled.json') 
trivia_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__trivia_to_trivia__test_subsampled.json')
squad_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__squad_to_squad__test_subsampled.json')
musique_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__musique_to_musique__test_subsampled.json')
hotpotqa_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json')
wikimultihopqa_multi_file = os.path.join("predictions", args.split, f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json')

nq_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__nq_to_nq__test_subsampled.json') 
trivia_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__trivia_to_trivia__test_subsampled.json')
squad_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__squad_to_squad__test_subsampled.json')
musique_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__musique_to_musique__test_subsampled.json')
hotpotqa_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json')
wikimultihopqa_one_file = os.path.join("predictions", args.split, f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json')

nq_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_nq____prompt_set_1', 'prediction__nq_to_nq__test_subsampled.json') 
trivia_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_trivia____prompt_set_1', 'prediction__trivia_to_trivia__test_subsampled.json')
squad_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_squad____prompt_set_1', 'prediction__squad_to_squad__test_subsampled.json')
musique_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_musique____prompt_set_1', 'prediction__musique_to_musique__test_subsampled.json')
hotpotqa_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_hotpotqa____prompt_set_1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json')
wikimultihopqa_zero_file = os.path.join("predictions", args.split, f'nor_qa_{args.model_name}_2wikimultihopqa____prompt_set_1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json')

dataName_to_multi_one_zero_file = {
    'musique' : {
        'C' : musique_multi_file,
        'B' : musique_one_file,
        'A' : musique_zero_file
    },
    'hotpotqa' : {
        'C' : hotpotqa_multi_file,
        'B' : hotpotqa_one_file,
        'A' : hotpotqa_zero_file
    },
    '2wikimultihopqa' : {
        'C' : wikimultihopqa_multi_file,
        'B' : wikimultihopqa_one_file,
        'A' : wikimultihopqa_zero_file
    },
    'nq' : {
        'C' : nq_multi_file,
        'B' : nq_one_file,
        'A' : nq_zero_file
    },
    'trivia' : {
        'C' : trivia_multi_file,
        'B' : trivia_one_file,
        'A' : trivia_zero_file
    },
    'squad' : {
        'C' : squad_multi_file,
        'B' : squad_one_file,
        'A' : squad_zero_file
    },
}

    
total_qid_to_classification_pred = load_json(classification_result_file)

for data_name in dataName_to_multi_one_zero_file.keys():
    save_prediction_with_classified_label(total_qid_to_classification_pred, data_name, stepNum_result_file, dataName_to_multi_one_zero_file, output_path)


