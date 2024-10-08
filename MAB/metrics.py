"""
calculate the metrics for the classifier 
"""
import json
import os
import sys

from sklearn.metrics import precision_score, recall_score, f1_score



from sklearn.metrics import classification_report
import numpy as np


import pdb

def compute_f1(y_pred, y_true):


    # Compute and display precision, recall, and F1-score for each class
    report = classification_report(y_true, y_pred, labels=['A', 'B', 'C'], target_names=['Label A', 'Label B', 'Label C'])
    print(report)
    # Assuming y_true and y_pred are defined
    results = []
    for average in ['micro', 'macro']:
        results.append({
        f'P ({average})':precision_score(y_true, y_pred, average=average),
        f'R ({average})': recall_score(y_true, y_pred, average=average),
        f'F1 ({average})':f1_score(y_true, y_pred, average=average),
        })
    results.append({
        "each class":report
    })
    return results

def calculate_acc(gold_label,pred_label):

   

    correct_count = 0


    for i in range(len(gold_label)):
        if gold_label[i] == pred_label[i]:
            correct_count += 1
        
    
    print(f"acc:{correct_count/len(gold_label)}")
    return correct_count/len(gold_label)



def main(pred_results_dir:str):

    # gold_label = json.load(open('/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/valid.json','r',encoding='utf-8'))
    # gold_label_dict = {item['id']:item for item in gold_label}

    # pred_label = json.load(open('/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-08-29-19-32-25_test','r',encoding='utf-8'))
   
    # pred_results = json.load(open(pred_results_dir,'r',encoding='utf-8'))

    single_hop = json.load(open('/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-09-02-11-58-20_test/dict_id_pred_results.json','r',encoding='utf-8'))

    multi_hop = json.load(open('/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-08-29-19-32-25_test/dict_id_pred_results.json','r',encoding='utf-8'))

    gold_label = []
    pred_label = []

    result = []

    gold_label_B_count = 0
    gold_label_C_count = 0
    for id,data in single_hop.items():
        # if data['dataset_name'] in ['squad','trivia','nq']:
            if  not data.get('gt_ans',None) is None:
                gold_label.append(data.get('gt_ans'))

                
                pred_label.append(data.get('prediction'))

                result.append({
                    "id":id,
                    "dataset_name":data['dataset_name'],
                    "gt_ans":data.get('gt_ans'),
                    "prediction":pred_label[-1]
                })

    # for id,data in single_hop.items():
    #     if data['dataset_name'] not in ['squad','trivia','nq']:
    #         if  not data.get('gt_ans',None) is None:
    #             gold_label.append(data.get('gt_ans'))
    #             pred_label.append(data.get('prediction'))
    #             result.append({
    #                 "id":id,
    #                 "dataset_name":data['dataset_name'],
    #                 "gt_ans":data.get('gt_ans'),
    #                 "prediction":data.get('prediction')
    #             })

    # print(gold_label)
    # print(pred_label)

    with open("./gold_label.json","w",encoding='utf-8') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=4))
    draw_confusion_matrix(gold_label,pred_label)

    # acc= calculate_acc(gold_label,pred_label)
    # f1_results = compute_f1(pred_label,gold_label)
    # f1_results.append({
    #     "acc":acc
    # })
    # # print(f1_results) 
    # output_path = os.path.dirname(pred_results_dir)+'/acc_f1_metrics.json'
    # print(output_path)
    # with open(os.path.dirname(pred_results_dir)+'/acc_f1_metrics.json','w',encoding='utf-8') as f:
    #     json.dump(f1_results,f,ensure_ascii=False,indent=4)
    #     f.close()

def draw_confusion_matrix(gold_labels,pred_labels):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 计算混淆矩阵
    cm = confusion_matrix(gold_labels, pred_labels)

    # 打印混淆矩阵
    print(cm)

        # 计算混淆矩阵的百分比形式
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 打印标准化后的混淆矩阵
    print(cm_normalized)

    # # 示例混淆矩阵数据
    # confusion_matrix = np.array([[0.31, 0.47, 0.22],
    #                             [0.1, 0.66, 0.23],
    #                             [0.03, 0.31, 0.65]])
    # 设置标签
    labels = ['No-Retrieval', 'Single-Step', 'Multi-Step']

    # 创建一个热图来表示混淆矩阵

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'normal'
    plt.figure(figsize=(10, 7))  # 设置图像大小
    ax = sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt=".2f", xticklabels=labels, yticklabels=labels)

    # 设置标题和坐标轴标签
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    
    # plt.colorbar(ax.collections[0])

  
    plt.show()
    plt.savefig('./confusion_matrix.pdf')

def draw():
    import matplotlib.pyplot as plt

    # Data for plotting
    categories = ['Adaptive Retrieval', 'Self-RAG', 'Adaptive-RAG', 'MBA-RAG']
    values = [42, 41.5, 54, 56.12]

    # Create the bar graph
    # title_font = {'family': 'Arial', 'color': 'darkred', 'weight': 'normal', 'size': 12}
    # label_font = {'family': 'Arial', 'color': 'blue', 'weight': 'normal', 'size': 12}
    # plt.rcParams['font.family'] = 'Arial'  # e.g., 'Arial', 'Helvetica', 'sans-serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'normal'  # e.g., 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'


    plt.figure(figsize=(8, 6))

    bars = plt.bar(categories, values, color=['#add8e6', '#A9D18E', '#dda0dd', '#ffcc99'],width=0.9)

    # Set axis ranges
    plt.ylim(35, 58)  # Setting the y-axis range from 0 to 20
    plt.yticks(np.arange(40, 56, 5))  # Optional: Setting the y-axis values
    plt.xlim(-0.8, len(categories)-0.2)  # Optional: Adjust the x-axis limits to tightly fit the bars

    # Adding labels and title
    # plt.xlabel('Classifier Acc')
    # plt.ylabel('Values')
    plt.title('Classifier Acc')

    # Optional: Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

    # Show the plot
    
    plt.show()
    plt.savefig('./acc.pdf')

if __name__ == '__main__':
    pred_results_dir = '/apdcephfs_cq10/share_1567347/share_info/zivtang/code/Adaptive-RAG/MAB/results1/2024-08-29-19-32-25_test/dict_id_pred_results.json'
    main(pred_results_dir)
    # draw()