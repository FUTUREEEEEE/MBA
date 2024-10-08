#parameters
lrs="0.00005"
epochs="1"
warmup_ratios="0"
accumulation_steps="1"
explore_rates="0.1"
schedulers="linear cosine"

# #parameters
# lrs="0.00003"
# epochs="1"
# warmup_ratios="0.05"
# accumulation_steps="1"
# explore_rates="0.1"
# schedulers="linear cosine"

random_gpu=$((RANDOM % 8))
echo "random_gpu: $random_gpu"

echo "MAB MO training start"
now=$(date +"%Y-%m-%d-%H-%M-%S")
echo $now
exp_name=test

python3 train_mab_mo_single.py \
    --exp_name test \
    --now $now \
    --lr $lrs\
    --epochs $epochs\
    --warmup_ratio $warmup_ratios\
    --accumulation_steps $accumulation_steps\
    --explore_rate $explore_rates\
    --out_dir ./results1/ \
    --debug \
    --device cuda:$random_gpu \
    # --use_binary  
    # --reward_zero 1  \
    # --reward_one 1 \
    # --reward_multiple 1 \
    # --skip_dataset musique,hotpotqa,2wikimultihopqa
    

echo "MAB MO training done"
cd .. # go to Adaptive-RAG
echo "predict complexity start"
python3 classifier/postprocess/predict_complexity_on_classification_results.py \
    --file_path MAB/results1/${now}_${exp_name}/dict_id_pred_results.json \
    --output_path /apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/${now}_${exp_name}

echo "predict complexity done"


echo "MAB MO evaluate start"
python3 evaluate_final_acc.py \
    --model_name MAB \
    --base_pred_path /apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/${now}_${exp_name}/
echo "MAB MO evaluate done"
