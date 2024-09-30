# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

class CustomTrainer(Trainer):
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        return data_collator

    def compute_loss(self, model, inputs, return_outputs=False):
        reward_weights = inputs.pop("reward_weights")
        for key in inputs.keys():
            num_of_samples = inputs[key].shape[0]
            break
        sample_loss = []
        sample_token = []
        for i in range(num_of_samples):
            sample = {}
            for key in inputs.keys():
                sample[key] = inputs[key][i:i+1]
            num_of_tokens = sum(inputs['attention_mask'][i])
            if return_outputs: 
                loss, outputs = super().compute_loss(model, sample, return_outputs)
            else:
                loss = super().compute_loss(model, sample, return_outputs)
            sample_loss.append(loss)
            sample_token.append(len(inputs['attention_mask'][i]))
        log = {
            "sample_loss": sample_loss
        }
        sample_loss_tensor = torch.stack(sample_loss).to(reward_weights.device)
        reward_weights_reshaped = reward_weights.view(-1)
        weighted_loss = sample_loss_tensor * reward_weights_reshaped
        average_weighted_loss = weighted_loss.mean()

        return (average_weighted_loss, outputs) if return_outputs else average_weighted_loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
    conv = get_conversation_template("zephyr")

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    total_num_of_data = len(sources)
    valid_num_of_data = 0

    # Apply prompt templates
    conversations = []
    losses = []
    for i, source in enumerate(sources):
        conv.system_message = source[0]["value"].strip()
        source = source[1:]

        conv.messages = []
        losses_for_one_interaction = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[(j+1) % 2], f"{i}"
            conv.append_message(role, sentence["value"])
            losses_for_one_interaction.append(sentence["loss"])
        conversations.append(conv.get_prompt())
        losses.append(losses_for_one_interaction)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    reward_weights = None

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.roles[1] + "\n"
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split("<|user|>\n")
        cur_len = 1  # for <s> special character
        target[:cur_len] = IGNORE_TOKEN_ID

        for i, turn in enumerate(turns):
            if turn == "":
                break

            if i != 0:
                turn = f"<|user|>\n{turn}"
            parts = turn.split(sep)  # user text and assistant text
            if len(parts) != 2:
                break
            parts[0] += sep
            turn_len = len(tokenizer(turn).input_ids) - 1  # loại bỏ kí tự <s>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            if sources[idx][2*i+1]["loss"] == 0.0:
                target[cur_len + instruction_len : cur_len + turn_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if reward_weights == None:
            reward_weights = [[sources[idx][-1]["loss"]]]
        else:
            reward_weights.append([sources[idx][-1]["loss"]])

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )
            else:
                valid_num_of_data += 1

    print(f"Valid data portion: {valid_num_of_data}/{total_num_of_data}")

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        reward_weights=torch.tensor(reward_weights)
    )

# def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
#     conv = get_conversation_template("zephyr")
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
#     total_num_of_data = len(sources)
#     valid_num_of_data = 0
#     # Apply prompt templates
#     conversations = []
#     losses = []

#     for i, source in enumerate(sources):
#         conv.system_message = source[0]["value"].strip()

#         # if roles[source[0]["from"]] != conv.roles[0]:
#         source = source[1:]

#         conv.messages = []
#         losses_for_one_interaction = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#             losses_for_one_interaction.append(sentence["loss"])
#         conversations.append(conv.get_prompt())
#         losses.append(losses_for_one_interaction)
    


#     # Tokenize conversations
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     ).input_ids

#     targets = input_ids.clone()

#     reward_weights = None

#     # Mask targets. Only compute loss on the assistant outputs.
#     sep = conv.roles[1] + "\n"
#     for idx, (conversation, target) in enumerate(zip(conversations, targets)):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())
#         turns = conversation.split("<|user|>\n")


#         for i in range(len(turns)):
#             print(f"turn{i}: {turns[i]}")
        
#         if turns[0] == "":
#             turns = turns[1:]
#         if turns[-1] == "":
#             turns = turns[:-1]
            
#         cur_len = 1  # for <s> special character

#         target[:cur_len] = IGNORE_TOKEN_ID
        
#         for i, turn in enumerate(turns):
#             # if turn == "":
#             #     break

#             # if i == 0:  # system message
#             #     parts = [turn, ""]
#             # else:
#             turn = f"<|user|>\n{turn}"
#             parts = turn.split(sep)  # user text and assistant text

#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             turn_len = len(tokenizer(turn).input_ids) - 1  # loại bỏ kí tự <s>
#             instruction_len = len(tokenizer(parts[0]).input_ids) - 1
#             target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
#             # print(f"length: {len(sources[idx])}")
#             if sources[idx][2*i+1]["loss"] == 0.0:
#                 target[cur_len + instruction_len : cur_len + turn_len] = IGNORE_TOKEN_ID
#             cur_len += turn_len

#         target[cur_len:] = IGNORE_TOKEN_ID
#         if reward_weights == None:
#             reward_weights = [[sources[idx][-1]["loss"]]]
#         else:
#             reward_weights.append([sources[idx][-1]["loss"]])
#         if False:  # Inspect and check the correctness of masking
#             z = target.clone()
#             z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#             rank0_print(tokenizer.decode(z))
#             exit()

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_TOKEN_ID
#                 rank0_print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" #turn = {len(turns) - 1}. (ignored)"
#                 )
#             else:
#                 valid_num_of_data += 1

#     print(f"Valid data portion: {valid_num_of_data}/{total_num_of_data}")

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#         reward_weights=torch.tensor(reward_weights)
#     )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.reward_weights = data_dict["reward_weights"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            reward_weights=self.reward_weights[i]
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            reward_weights=ret["reward_weights"][0]
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    
    if tokenizer.pad_token is None:
        print(f"Adding pad token as '<pad>'")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>"),
            tokenizer=tokenizer,
            model=model,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    # tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    # Start trainner
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
