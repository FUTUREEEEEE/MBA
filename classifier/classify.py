import torch


from torch import sigmoid
from typing import Optional

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence,List
import logging
import os,sys

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig

from datasets import load_dataset
import datasets
import numpy as np
# from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


from datasets import load_dataset
from torch.utils.data import Dataset


import pandas
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score

import pdb

# from trainer_custom import Trainer

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
# IGNORE_TOKEN_ID = LabelSmoother.ignore_index



def read_jsonline(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def jsonline_load(f, mode="r"):
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    json_objects_list = [json.loads(line) for line in f]
    return json_objects_list

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    # print(preds_[0].shape,preds_[1].shape)
    true_labels_np = prediction.label_ids
    # print(true_labels_np.shape)
    pred_labels_np  = (sigmod(preds_[0]) >= 0.5).astype(int)

    # print(f"true_labels_np:\n{true_labels_np}\n,pred_labels_np:\n{pred_labels_np}")
    
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
        'f1_macro': f1_macro
    }


def compute_metrics(prediction):
    predictions = prediction.predictions
    labels_ = prediction.label_ids

   

    classes=[0,1,2]

    # Converting logits to predicted labels
    preds_ = np.argmax(predictions, axis=1)
    

    accuracy = (preds_ == labels_).mean()

    # One-hot encoding of predictions and labels
    labels = label_binarize(labels_, classes=classes)
    preds = label_binarize(preds_, classes=classes)

    # Compute detailed metrics per class
    results = {}
    results['accuracy'] = accuracy
    for i, class_label in enumerate(classes):
        
        precision = precision_score_custom(preds[:, i], labels[:, i])
        recall = recall_score_custom(preds[:, i], labels[:, i])
        f1 = f1_score_custom(preds[:, i], labels[:, i])
        results[str(class_label)] = {"precision": precision, "recall": recall, "f1": f1}#加str是因为,Trainer内部是有一个前缀验证

    print_rank_0(f"labels_:\n{labels_}\n")
    print_rank_0(f"preds_:\n{preds_}\n")
    print_rank_0(f"labels.shape:{labels_.size}, preds_.size:{preds_.size}")

    return results


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



def _tokenize_fn(strings: Sequence[str],labels: Sequence[List[int]], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens  = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]

   
    
    # labels = [int(label) for label in labels]
   

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    # targets: Sequence[str],
    targets: Sequence[List[int]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_tokenized = _tokenize_fn(sources, targets,tokenizer)
    input_ids = sources_tokenized["input_ids"]
    labels = sources_tokenized['labels']

    
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labels = [torch.tensor(x) for x in labels]
        # labels = [torch.tensor(x) for x in labels]

        labels_tensor = torch.zeros((len(labels), 3))  # 3个标签，初始化为全零张量
        for i, label_list in enumerate(labels):
            labels_tensor[i, label_list] = 1  # 对于每个样本，设置相应的标签位置为1
        
        
        
        return dict(
            input_ids=input_ids,
            labels=labels_tensor,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        
       
        
      
class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, format_mode:str="t5",data_type="train") -> None:
        super(SupervisedDataset).__init__()

        print_rank_0("Loading data...")
        # list_data_dict = utils.jload(data_path)
        
        list_data_dict = read_jsonline(data_path)
        print(f"len list_data_dict:{len(list_data_dict)}")
        self.input_ids = []
        self.labels = []
        # list_data_dict = pandas.read_excel(data_path).to_dict(orient="records")
        # print_rank_0("Length of training data: "+str(len(list_data_dict)))

        if data_type=="train":
            random.shuffle(list_data_dict)
        if format_mode=="qwen2":
            pass
        elif format_mode=="t5":
            sources = []
            targets = []

            for data in list_data_dict:
                sources.append(data['inputs'])
                targets.append(data['labels'])

            print_rank_0(f"intput:{sources[0]}\nlable:{targets[0]}\n{type(sources)},{len(sources)}")
            data_dict = preprocess(sources, targets, tokenizer)
            
            self.input_ids = data_dict['input_ids']
            self.labels = data_dict['labels']


     #要构造类方法能取元素
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])






def print_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

@dataclass
class ModelArguments:
    lora_trainable : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default="embed_tokens,lm_head")
    use_lora : Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-moe-16b")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."}),
    eval_path: str = field(default=None, metadata={"help": "Path to the eval data."}),
    eval_output_path: str = field(default=None, metadata={"help": "Path to the eval data."}),


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    ),
    do_train: Optional[bool] = field(default=True),
    do_eval: Optional[bool] = field(default=True),
    
    padding_side:Optional[str] = field(default="right")

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def build_model(model_args, training_args, checkpoint_dir):
    if not model_args.use_lora: assert model_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float16)


    model = transformers.T5ForSequenceClassification.from_pretrained(
    # model = transformers.DistilBertForSequenceClassification.from_pretrained(
    # model = LlamaMoEForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        # use_cache=False,
        # output_router_logits=True,##注意添加：计算router的loss
    )

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32

    model.config.problem_type = "multi_label_classification"
    # model.config.num_labels =3


    if model.config.pad_token_id==None:
        # model.config.pad_token = model.config.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    

    try:
        model.print_trainable_parameters()
    except Exception:
        print_trainable_parameters(model)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
    return model
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)
    
    # from llama.tokenization_llama import LlamaTokenizer
    # from modeling_file.qwen2_moe.configuration_qwen2_moe
    # from modeling_file.llama3_moe.tokenization_llama_fast import LlamaTokenizerFast

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    # tokenizer = LlamaTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=training_args.padding_side,
        use_fast=False,
        trust_remote_code=True
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    # if tokenizer.bos_token is None:
    #      tokenizer.bos_token = "<|im_start|>"
    #      tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         tokenizer.pad_token_id = tokenizer.eos_token_id

        

    if training_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(model_args.model_name_or_path))
    
   
    
    # raw_train_datasets = load_dataset(
    #     'json',
    #     data_files=data_args.data_path,
    #     split="train",
    #     cache_dir=training_args.cache_dir,
    #     # cache_dir=None
    #     # cache_dir=None
    #     )
    
    # raw_train_datasets.shuffle(seed=42)


    # if training_args.local_rank > 0:
    #     torch.distributed.barrier()
        
    # train_dataset = raw_train_datasets.map(
    #     train_tokenize_function,
    #     batched=True,
    #     batch_size=1000,
    #     num_proc=5,
    #     remove_columns=raw_train_datasets.column_names,
    #     load_from_cache_file=True, # not args.overwrite_cache
    #     desc="Running Encoding",
    #     fn_kwargs={ "tokenizer": tokenizer }
    # )
    

    if training_args.local_rank>0:
        torch.distributed.barrier()
   
    train_dataset = SupervisedDataset(data_args.data_path,tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(data_args.eval_path,tokenizer=tokenizer,data_type="eval")

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    print_rank_0(f"Training dataset samples:{len(train_dataset)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir)
    
    print_rank_0(f"\n\n****resulme from :{resume_from_checkpoint_dir}****\n\n")
    model = build_model(model_args, training_args, resume_from_checkpoint_dir)

    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics_,**data_module)
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)

    
    if training_args.do_train:
        logger.info("*** Training ***")
        trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
        trainer.save_state()
        

        trainer.save_model(output_dir=training_args.output_dir)
        tokenizer.save_pretrained(save_directory=training_args.output_dir)
    
        # Evaluation
    if training_args.do_eval:
        model.eval()
        logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate()
        output = trainer.predict(eval_dataset)
        print(f"****************output**************************:{output}")
        # print(f"metrics:\n\n{json.dumps(metrics,indent=4)}\n\n")


        
        output = trainer.predict(test_dataset=None)
        # metrics = output.metrics


       

        """ import math
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity """

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        model.eval()
        logger.info("*** Predict ***")
        output = trainer.predict(eval_dataset)
        predictions = output.predictions
        print(f"****************output**************************:{output}")


if __name__ == "__main__":
    train()
