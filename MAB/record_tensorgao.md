# tensorgao 修改记录
 8.25待完成
 1. 整合classifier/postprocess/predict_complexity_on_classification_results.py 与evaluate_final_acc.py MAB/train_mab_mo.py中，一步完成。注意添加mab的step结果

    train, valid 加step num
    加gold label. 给

    修改了classifier/postprocess/predict_complexity_on_classification_results.py， 先用已有的代替
        stepNum_result_file = "/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/predictions/test/ircot_qa_flan_t5_xl/total/stepNum.json"
    修改mab_mo.py中的exp_name和now参数，从train.sh中定义，后续可以访问到。
    ![alt text](image.png)
    ![alt text](image-1.png)

    distilbert:best 2024-08-29-19-32-25_test
    * /apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-08-30-20-10-29_test
        trival: 1-2 points high
             "em": 0.534,
                "f1": 0.621,
                "count": 500,
                "acc": 0.598

        squad: 0-1 points high
            "em": 0.272,
            "f1": 0.387,
            "count": 500,
            "acc": 0.334
        nq: same
             "em": 0.374,
            "f1": 0.476,
            "count": 500,
            "acc": 0.438

        musique: 0-1 points low
                "f1": 0.314,
                "em": 0.224,
                "count": 500,
                "acc": 0.25
        hotpotqa：1-2 points high
             "f1": 0.55283,
                "em": 0.436,
                "acc": 0.458
        2wiki: 6-8 points high
        "f1": 0.5761,
        "em": 0.482,
        "acc": 0.54
    /apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-09-02-11-45-37_test/eval_metic_result_acc_summary.json

    加上binary, -reward 调整

## 8.27 
    1. 添加新用户 user: elasticsearch, password: GQgq11223344
    2.安装jdk11 
    权限问题

    直接使用t5-xl的step

# xiaqiang 修改记录
1. dict.json save to results/xxx_exp_name dir for reproduction