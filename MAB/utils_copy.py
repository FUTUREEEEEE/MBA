import requests
import os
import json
# from SPARQLWrapper import SPARQLWrapper, JSON
import re
import argparse
import torch
import numpy as np
import random
import torch.nn.functional as F
import functools
import pandas as pd
import ast  
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_json(file_path):
    with open(file_path,'r',encoding='utf-8')as f:
        return json.load(f)

def write_json(file_path,data):
     with open(file_path,'w',encoding='utf-8')as f:
            json.dump(data,f,ensure_ascii=False,indent=4)

def softmax(data, tau=1.2):
    softm = np.exp(data/tau) / np.sum(np.exp(data/tau))
    return softm

def running_mean(data,window=50):
    if data is None or len(data) == 0:
        return []
    c = data.shape[0] - window
    smoothened = np.zeros(c)
    conv = np.ones(window)
    for i in range(c):
        smoothened[i] = (data[i:i+window] @ conv)/window
    return smoothened

def check_available_memory(device_id, required_memory_gb):
    """Check if the given GPU device has the required amount of free memory in GB."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    cached_memory = torch.cuda.memory_reserved(device_id)

    free_memory_gb = (total_memory - allocated_memory - cached_memory) / (1024**3)
    return free_memory_gb >= required_memory_gb


def select_gpu(required_memory_gb):
    """Select a GPU with at least the specified amount of free memory in GB, 
    sampling from available GPUs based on their free memory."""
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    free_memory_list = []
    for device_id in range(num_devices):
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        free_memory_gb = (total_memory - allocated_memory - cached_memory) / (1024**3)
        free_memory_list.append(free_memory_gb)

    # Filter out GPUs that don't meet the required memory, and use the remaining memory as weights for sampling
    weights = [free_memory if free_memory >= required_memory_gb else 0 for free_memory in free_memory_list]
    if sum(weights) == 0:
        raise RuntimeError("No suitable GPU found")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    # Randomly select a GPU based on the normalized weights
    device_id = random.choices(range(num_devices), weights=normalized_weights, k=1)[0]
    if not check_available_memory(device_id, required_memory_gb):
        raise RuntimeError("Selected GPU does not have enough free memory")
    
    return device_id

class Environment(object):
    
    _method_delay = {0:3,1:1,2:15} # 
    # _method_delay = {0:3,1:15} # *cwq


    
    method_delay_ce_label = F.softmax(torch.tensor(list(_method_delay.values()),dtype=torch.float ).reciprocal(),dim=0).detach().cpu().numpy()
    

    def __init__(self, arms, dataset,args,preding=False):
        self.arms = arms
        self.dataset = dataset
        self.preding = preding # 是否在make final prediction
        self.index = -1
        self._method_index =  {0:'zero',1:'one',2:'multiple'} #{0:'StructGPT',1:'Dal',2:'ToG',3:'BGE'} 
        # self._method_index =  {0:'StructGPT',1:'ToG',2:'BGE'}
        self.skip_dataset = args.skip_dataset

        #reward vallue
        self.reward_zero  =args.reward_zero
        self.reward_one = args.reward_one
        self.reward_multiple = args.reward_multiple



        self._update_state()
        

    def _update_state(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0
        
        while self.dataset[self.index]['dataset_name'] in self.skip_dataset and not self.preding:
           
            self.index += 1
            if self.index >= len(self.dataset):
                self.index = 0

        self.state = self.dataset[self.index]['question']
        
        # self.state = np.random.randint(0, self.arms)
    def _index_to_arm(self,index):
        if type(index) == np.ndarray:
            assert len(index) == 1
            index = index[0]
        return self._method_index[index]
        
    def get_state(self):
        return self.state
        # return self.state

    def _get_reward(self, arm):
        method = self._index_to_arm(arm)
        answer = self.dataset[self.index].get("total_answer",None) if self.dataset[self.index].get("total_answer",None) else self.dataset[self.index].get("answer",None)
    
        if method in answer:
            if method == 'zero':
                return self.reward_zero
            elif method == 'one':
                return self.reward_one
            elif method == 'multiple':
                return self.reward_multiple
               
            # return 1
        else:
            return -1
    
    def _get_recall(self,arm):
        raise NotImplementedError
        method = self._index_to_arm(arm)
        return self.dataset[self.index][method+'_eval']['recall']
    
    
    def get_delay(self,arm):
        if type(arm) == np.ndarray:
            assert len(arm) == 1
            arm = arm[0]
            
        return torch.tensor(self._method_delay[arm])

    def choose_arm(self, arm):
        reward = self._get_reward(arm)
        # recall = self._get_recall(arm)
        self._update_state()
        return reward
    
    def __len__(self):
        return len(self.dataset)