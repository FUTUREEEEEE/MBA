import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_scheduler
import concurrent.futures
from functools import partial
from collections import defaultdict


from utils_copy import *

import multiprocessing

multiprocessing.set_start_method('spawn', force=True)


def running_mean(data,window=50):
    c = data.shape[0] - window
    smoothened = np.zeros(c)
    conv = np.ones(window)
    for i in range(c):
        smoothened[i] = (data[i:i+window] @ conv)/window
    return smoothened

def list_of_strings(arg):
	return arg.split(',')

def create_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the arguments
    parser.add_argument('--num_arms', type=int, default=3,
                        help='The string to process')
    parser.add_argument('--lr', type=float, default=1e-5,)
    parser.add_argument('--epochs', type=int, default=1,)
    parser.add_argument('--device', type=str, default='cuda:1',)
    parser.add_argument('--out_dir', type=str,default='./results',
                        help='The string to process')
    parser.add_argument('--exp_name', type=str,default="test")
    parser.add_argument('--des', type=str,required=False,default="")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--warmup_ratio',type=float,default=0)
    parser.add_argument('--MO',action='store_true')
    parser.add_argument('--accumulation_steps',type=int,default=1)
    parser.add_argument('--loss_policy',type=str,default='MSE')
    parser.add_argument('--action_policy',type=str,default='decay_greedy')
    parser.add_argument('--explore_rate',type=float,default=0.1)
    parser.add_argument('--online_learning',action='store_true')
    parser.add_argument('--now',type=str,default="")
    parser.add_argument('--model',type=str,default='bert')
    parser.add_argument('--reward_zero',type=float,default=1)
    parser.add_argument('--reward_one',type=float,default=3)
    parser.add_argument('--reward_multiple',type=float,default=4)
    parser.add_argument('--skip_dataset',type=list_of_strings,default=[])
    parser.add_argument('--use_binary',action='store_true')
    parser.add_argument('--generate_model_type',type=str,default='t5-xl',choices=['t5-xl','t5-xxl','gpt3.5'])



    return parser

def softmax(data, tau=1.2):
    softm = np.exp(data/tau) / np.sum(np.exp(data/tau))
    return softm




def test_network(environ, net,tokenizer, args):
    net.eval()
    if "bert" in args.model:
        net.to(args.device)
    rewards = []
    actions = []
    delays = []
    
    if args.online_learning:
        num_training_steps = args.epochs * len(environ) 
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
        
        optimizer = AdamW(net.parameters(), lr=args.lr)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )
    
    pbar = tqdm(total=len(environ), desc="Testing", dynamic_ncols=True)
    with torch.no_grad():
        for e in range(1, len(environ) + 1):
            query = environ.get_state()
            rewards_pred = net(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(args.device))
            action_probas = softmax(rewards_pred.logits[0].detach().cpu().numpy().copy())
            action = np.argmax(action_probas)
            # action = np.random.choice(args.num_arms, p=action_probas)
            reward = environ.choose_arm(action)
            
            if args.online_learning:
                # update model
                net.train()
                loss = policy_gradient_loss(args.loss_policy,reward,action,rewards_pred,criterion)
                optimizer.zero_grad()
                loss.backward()
                lr_scheduler.step()
                optimizer.step()
                net.eval()
                
            
            if args.MO:
                delay = environ.get_delay(action)
                delays.append(delay)
            
            actions.append(action)
            
            if reward == -1:
                reward = 0
            if reward > 0:
                reward = 1
                
            rewards.append(reward)
            
            pbar.update(1)
    
    # count each action
    action_count = np.zeros(args.num_arms)
    for i in actions:
        action_count[i] += 1
    print(f"Action count : {action_count}")  
    if args.MO:
        print(f"Average delay : {np.mean(delays)}")      
    
    return np.array(rewards),actions,delays


def predict_network(environ, net,tokenizer, args):
    net.eval()
    if "bert" in args.model:
        net.to(args.device)

    rewards = []
    actions = []

    
    pbar = tqdm(total=len(environ), desc="Predicting", dynamic_ncols=True)
    with torch.no_grad():
        for e in range(1, len(environ) + 1):
            query = environ.get_state()
            rewards_pred = net(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(args.device))
            action_probas = softmax(rewards_pred.logits[0].detach().cpu().numpy().copy())
            action = np.argmax(action_probas)

            environ._update_state()
            
            actions.append(action)
            
            pbar.update(1)
    
    # count each action
    action_count = np.zeros(args.num_arms)
    for i in actions:
        action_count[i] += 1
    print(f"Action count : {action_count}")  
    
    return actions

def policy_gradient_loss(policy,reward,arm,rewards_pred,criterion=None):
    if policy == 'MSE':
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward
        loss = criterion(rewards_pred.logits[0].cpu(), torch.Tensor(true_rewards)) 
    
    elif policy == 'grad_ascent':
        if reward <= 0:
            loss = F.softmax(rewards_pred.logits[0],dim=0)[arm] 
        else:
            loss = - F.softmax(rewards_pred.logits[0],dim=0)[arm]
            # loss = - F.softmax(rewards_pred.logits[0],dim=0).clone()[arm]

            
        if reward == -1:
            reward = 0
            
            
    if policy == 'MSE_GGC':
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward
        loss = criterion(rewards_pred.logits[0].cpu(), torch.Tensor(true_rewards)) 
        
    
    return loss,reward
    
    

def train_network(environ, net,tokenizer, args):
    # optimizer and scheduler
    
    accumulated_loss = 0.0  # 新增：用于累积损失
    accumulation_steps = args.accumulation_steps  # 梯度累积步骤（等于batch_size）
    
    num_training_steps = args.epochs * len(environ) 
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    if args.loss_policy == 'MSE' or args.loss_policy == 'MSE_GGC':
        optimizer = AdamW(net.parameters(), lr=args.lr)
        lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps// accumulation_steps
        )
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
        lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps// accumulation_steps
        )


    
    criterion = nn.MSELoss()
    
    
    # * net init
    net.train()
    if "bert" in args.model:
        net.to(args.device)
    

    
    # * variables init
    rewards = []
    losses = []
    weight_changes = []
    pbar = tqdm(total=num_training_steps, desc="Training", dynamic_ncols=True)
    action_probas_list = []


    
    previous_weights = [param.data.clone() for param in net.parameters()]

    for e in range(1, num_training_steps + 1):
        query = environ.get_state()
        rewards_pred = net(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(args.device))
        action_probas = softmax(rewards_pred.logits[0].detach().cpu().numpy().copy())
        action_probas_list.append(action_probas)
        
        # * action policy
        if args.action_policy == 'greedy' and np.random.rand() < args.explore_rate:
            arm = np.random.randint(0, args.num_arms)
        elif args.action_policy == 'decay_greedy' and np.random.rand() < float(1/ np.log(e + 0.00001)):
            arm = np.random.randint(0, args.num_arms)
        else:
            arm = np.argmax(action_probas)
            
        
        reward = environ.choose_arm(arm)
        
        # * loss policy

        loss,reward = policy_gradient_loss(args.loss_policy,reward,arm,rewards_pred,criterion)

        
        loss = loss/accumulation_steps
        losses.append(loss.item())
        if reward>0:
            reward = 1
        else:
            reward = 0
        rewards.append(reward)
        

        # * gradient accumulation
        loss.backward()  # 反向传播，但不立即更新权重
        accumulated_loss += loss.item()

        if e % accumulation_steps == 0 or e == num_training_steps:
            optimizer.step()  # 更新权重
            lr_scheduler.step()
            optimizer.zero_grad()  # 重置梯度

            pbar.set_postfix({'loss': round(accumulated_loss / accumulation_steps, 2)}, refresh=True)
            accumulated_loss = 0.0  # 重置累积损失
            
            # 计算和记录权重的变化
            weight_change = [torch.norm(param.data - prev_param).item() for param, prev_param in zip(net.parameters(), previous_weights)]
            weight_changes.append(weight_change)
            # 更新 previous_weights 以用于下一次迭代
            previous_weights = [param.data.clone() for param in net.parameters()]


        pbar.update(1)


    return (np.array(losses), np.array(rewards), np.array(action_probas_list),np.array(weight_changes))

def load_model(args):

    if args.model == 'bert-large':
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-large-uncased')


        model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-large-uncased',
                                                                num_labels=args.num_arms)
    elif args.model == 'distilbert':
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


        model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased",
                                                                num_labels=args.num_arms)
    else:
        raise ValueError("Model not supported")
    return tokenizer, model

def one_experiment(args):
    fix_random_seed(args.seed)

    assert args.debug, "should run in debug mode"
    
    
    train_set = read_json('train.json')



    print(args.use_binary)


    train_set = np.random.permutation(train_set)

    tokenizer, model = load_model(args)
    env = Environment(arms=args.num_arms,dataset=train_set,args=args)
    
    losses, rewards, action_probas, weight_changes = train_network(env, model,tokenizer,args)
    
    test_set = read_json('valid.json')
    env = Environment(arms=args.num_arms,dataset=test_set,args=args)
    
    test_rewards,test_actions,test_delays = test_network(env, model,tokenizer,args)
    print(f"Test Reward : {np.mean(test_rewards)}")

    pred_set = read_json('predict.json')
    
    env = Environment(arms=args.num_arms,dataset=pred_set,args=args,preding=True)

    
    pred_actions = predict_network(env, model,tokenizer,args)

    # make dict_id_pred_results.json
    pred_results = {}   
    mapping = {0:'A',1:'B',2:'C'}
    for i in range(len(pred_actions)):
        pred_results[pred_set[i]['id']] = {
            "dataset_name":pred_set[i]['dataset_name'],
            "question":pred_set[i]['question'],
            "prediction":mapping[pred_actions[i]],
            "gt_ans":pred_set[i].get("gt_ans",None),
            "gt_multi_ans":pred_set[i].get("gt_multi_ans",None)
            }

    # * save 
    write_json(f"{args.out_dir}/{now}_{args.exp_name}/dict_id_pred_results.json",pred_results)
    # touch a np.mean(test_rewards):.2f.txt
    with open(f"{args.out_dir}/{now}_{args.exp_name}/test_reward_{np.mean(test_rewards):.2f}.txt", "w") as f:
        f.write("")
    
    # calcualte distribution of gt_ans，分开统计有gold_ans和没有的样本
    Has_gt_ans_gold_smaple_distribution = defaultdict(int)
    No_gt_ans_smaple_distribution = defaultdict(int)
    Has_gt_ans_pred_smaple_distribution = defaultdict(int)

    for id,data in enumerate(pred_results):
        if pred_results[data]['gt_ans'] != None:
            Has_gt_ans_gold_smaple_distribution[pred_results[data]['gt_ans']] += 1
            Has_gt_ans_pred_smaple_distribution[pred_results[data]['prediction']] += 1
        else:
            No_gt_ans_smaple_distribution[pred_results[data]['prediction']] += 1
    
    total_distribution={
        "Has_gt_ans_gold_smaple_distribution":Has_gt_ans_gold_smaple_distribution,
        "Has_gt_ans_pred_smaple_distribution":Has_gt_ans_pred_smaple_distribution,
        "no_gt_ans_sample_distribution":No_gt_ans_smaple_distribution,
        
    }

    write_json(f"{args.out_dir}/{now}_{args.exp_name}/smaple_distribution.json",total_distribution)

    test_set_info = {} #* leave empty for now

    return env,losses,rewards,action_probas,weight_changes,test_set_info,test_rewards,test_actions,test_delays

def run(seed,args,now):

    args.seed = seed
    # args.now = now
        
    args.device = select_gpu(10) # random free gpu with 20 gb memory

    test_env, train_losses, train_rewards, train_action_probas, weight_changes, test_set_info, test_rewards, test_actions,test_delays = one_experiment(args)
    
    result = {
        "test_rewards": test_rewards,
        "train_rewards": train_rewards,
        "test_actions": test_actions,
        "method_index": test_env._method_index,
        "test_set_info": test_set_info,
        "test_delays":test_delays
    }
    
    # * plt 
    plt.figure(figsize=(10, 12))
    plt.subplot(4, 1, 1)
    for i in range(train_action_probas.shape[1]):  # Looping over the number of actions
        plt.plot(running_mean(train_action_probas[:, i], 50), label=f'Action {i+1} Probability')
    plt.legend()
    plt.title('Action Probabilities')

    # Second subplot for average reward
    plt.subplot(4, 1, 2)
    plt.plot(running_mean(train_rewards, window=50), label="Train Reward")
    plt.plot(running_mean(test_rewards, window=50), label="Test Reward")
    plt.legend()
    plt.title('Average Reward')

    # Third subplot for train loss
    plt.subplot(4, 1, 3)
    plt.plot(running_mean(train_losses, 50), label="Train Loss")
    plt.legend()
    plt.title('Train Loss')
    
    # Fourth subplot for weight changes
    plt.subplot(4, 1, 4)
    plt.plot(weight_changes)
    # plt.legend()
    plt.title('Weight Changes')
    
    # get string date time 
    
    plt.tight_layout()
        
    # if args.debug:
    #     plt.savefig(f"results/{now}_{args.exp_name}_debug.png")
    # else:
    #     plt.savefig(os.path.join(args.out_dir,now + "_" + args.exp_name, f'_train_mab{seed}.png'))
        
    # Collect all necessary information in a dictionary


    return result


if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    print(args)


    # assert args.lr < 1e-4, "lr should be smaller than 1e-5" 
    
    if args.now != "":
        now = args.now
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    
    # create output directory
    os.mkdir(os.path.join(args.out_dir,now + "_" + args.exp_name))
     # Convert the arguments to a dictionary
    args_dict = vars(args)
    write_json(os.path.join(args.out_dir,now + "_" + args.exp_name,"args.json"),args_dict)
    # save current script
    os.system(f"cp {__file__} {args.out_dir + now + '_' + args.exp_name + '/'}")
    
    avg_test_rewards = []
    avg_train_rewards = []
    avg_test_delays = []
    
    if args.debug:
        seeds = args.seed
        results = [run(args.seed,args,now)]
    else:    
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50,51]
        
        partial_run = partial(run, args=args,now=now)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(partial_run, seeds))
        
    # for result in results:
    #     avg_test_rewards.append(np.mean(result["test_rewards"]))
    #     avg_train_rewards.append(np.mean(result["train_rewards"]))
    #     if args.MO:
    #         avg_test_delays.append(np.mean(result["test_delays"]))
    
    # best_hit = -1
    # for i in results[0]['method_index'].values():
    #     if results[0]["test_set_info"][i]['hit'] > best_hit:
    #         best_hit = results[0]["test_set_info"][i]['hit']
    #         best_method = i
    
    # print(f"Best method : {best_method} with hit {best_hit:.3f} ")  
    # print(f"Average test reward : {np.mean(avg_test_rewards)}")      
    
    # if not args.debug:
    #     #* write experiment log 
    #     with open(args.out_dir + 'log.txt', 'a') as f:
    #         f.write(f"Experiment name : {args.exp_name} \n")
    #         f.write(f"Experiment time : {now} \n")
    #         f.write(f"Experiment args : {args} \n")
    #         f.write(f"Experiment best method : {best_method} with hit {best_hit:.5f} \n")
    #         f.write(f"Experiment average test reward : {np.mean(avg_test_rewards):05f} ± {np.std(avg_test_rewards):05f} \n")
    #         f.write(f"Experiment best test reward : {np.max(avg_test_rewards):05f} \n")
    #         if args.MO:
    #             f.write(f"Experiment average test delay : {np.mean(avg_test_delays):05f} \n")
                
    #         f.write(f"Experiment average train reward : {np.mean(avg_train_rewards):05f} \n")
    #         f.write(f"Experiment Action count : {np.unique(results[0]['test_actions'], return_counts=True)} \n")
    #         # wirte config
    #         f.write(f"_method_index : {results[0]['method_index']} \n")
    #         f.write(f"plot : {args.out_dir + now + ' ' + args.exp_name + ' train_mab.png'} \n")
    #         f.write(f'Decription : {args.des} \n')
    #         f.write('------------------------------------\n')
        
