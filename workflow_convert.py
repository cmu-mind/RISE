import random
import csv
import argparse
import os
import pandas as pd
import numpy as np
import sys
import json
import copy
import matplotlib.pyplot as plt

from utils import DIALOGUES
from transformers import AutoTokenizer

tokenizer = None
env = None

def count_tokens(dialogue):
    global tokenizer
    prompt_token = 0
    response_token = 0
    for i in range(len(dialogue) - 1):
        message = dialogue[i]["value"]
        tokens = tokenizer.tokenize(message)
        prompt_token += len(tokens)
    
    message = dialogue[-1]["value"]
    tokens = tokenizer.tokenize(message)
    response_token += len(tokens)

    return prompt_token, response_token


def init_conversation(query):
    global env
    conversation = [
        {
            "from": "human",
            "value": DIALOGUES[env][0]["content"],
            "loss": 0.0,
        },
        {
            "from": "gpt",
            "value": DIALOGUES[env][1]["content"],
            "loss": 0.0
        },
        {
            "from": "human",
            "value": f"Great job! You've successfully answered the question. Now, let's move on to the next one.\n{query}",
            "loss": 0.0
        }
    ]
    return conversation

def create_user_message(observation):
    user_message = {
        "from": "human",
        "value": observation,
        "loss": 0.0
    }
    return user_message

def create_agent_message(action, reward):
    agent_message = {
        "from": "gpt",
        "value": action,
        "loss": reward
    }
    return agent_message


def skip_dialogue(only_success, log):
    max_reward = log["max_reward"]
    if only_success:
        return max_reward != 1.0
    else:
        return False
def remove_duplication(actions, observations, rewards):
    seen_actions = set()
    for i in range(len(actions)):
        actions[i] = actions[i].strip()
        if actions[i] in seen_actions or actions[i] == '':
            actions[i] = None
            observations[i] = None
            rewards[i] = None
        else:
            seen_actions.add(actions[i])
    actions = [action for action in actions if action != None]
    observations = [observation for observation in observations if observation != None]
    rewards = [reward for reward in rewards if reward != None]
    
    return actions, observations, rewards

def convert_to_data(log, args):
    dialogue = log["dialogue"]
    query = log["query"]
    turn_history = log["turn_history"]

    actions = turn_history["best_actions"]
    observations = turn_history["best_observations"]
    rewards = turn_history["best_rewards"]
    if args.remove_duplication:
        actions, observations, rewards = remove_duplication(actions, observations, rewards)

    dataset = []
    conversations = init_conversation(query)

    for i in range(len(actions)):
        action = actions[i]
        observation = observations[i]
        reward = rewards[i]
        action = actions[i]

        conversations.append(create_agent_message(action, reward))
        # if conversations[-1]["loss"] != 0.0:
        if len(conversations) > 2:
            if i < 1 and args.criteria != "success":
                dataset.append(copy.deepcopy(conversations))
            else:
                keep = True
                if args.criteria == "best":
                    for j in range(i):
                        if rewards[i] < rewards[j]:
                            keep = False
                            break
                elif args.criteria == "better":
                    if rewards[i] <= rewards[i-1]:
                        keep = False
                elif args.criteria == "success":
                    if rewards[i] != 1.0:
                        keep = False
                elif args.criteria == "all":
                    keep = True
                else:
                    print("undefine criteria")
                    exit()
                if keep:
                    dataset.append(copy.deepcopy(conversations))
        
        if conversations[-1]["loss"] == 1.0:
            break
        conversations[-1]["loss"] = 0.0
        conversations.append(create_user_message(observation))

    return dataset

def deep_copy_dataset(dataset):
    new_dataset = []
    for conversation in dataset:
        new_dataset.append(copy.deepcopy(conversation))
    return new_dataset

def exp_dataset_reward(dataset, temperature=1.0):
    for conversations in dataset:
        conversations[-1]["loss"] = np.exp(conversations[-1]["loss"]/temperature)
    return dataset

def plot_rewards_hist(rewards, save=None):
    plt.hist(rewards, bins=100)
    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()

def get_rewards(dataset):
    rewards = []
    for conversations in dataset:
        rewards.append(conversations[-1]["loss"])
    return rewards

def conversations_to_dataset(dataset):
    new_dataset = []
    index = 0
    for conversation in dataset:
        interaction = {
            "id": f"identity_{index}",
            "conversations": conversation,
        }
        new_dataset.append(interaction)
        index += 1
    return new_dataset

def save_dataset(dataset, filename):
    rewards = get_rewards(dataset)
    plot_rewards_hist(rewards, save=f"{filename}_hist.png")
    fine_tune_dataset = conversations_to_dataset(dataset)
    with open(f"{filename}", "w") as f:
        json.dump(fine_tune_dataset, f, indent=4)

def data_analysis(dataset):
    prompt_tokens = []
    response_tokens = []
    for dialogue in dataset:
        prompt_token, response_token = count_tokens(dialogue)
        prompt_tokens.append(prompt_token)
        response_tokens.append(response_token)
    size = len(dataset)
    prompt_mean = np.mean(np.array(prompt_tokens))
    prompt_var = np.var(np.array(prompt_tokens))
    response_mean = np.mean(np.array(response_tokens))
    response_var = np.var(np.array(response_tokens))

    print(f"Size of dataset: {size}")
    print(f"Prompt Length: [{prompt_mean}, {prompt_var}]")
    print(f"Response Length: [{response_mean}, {response_var}]")
    print(f"====================================================")
    print(f"")

def main():
    global tokenizer
    global env
    parser = argparse.ArgumentParser(description='Convert dialogue into a datast for SFT')
    parser.add_argument('--filepath', type=str, help='path to log')
    parser.add_argument('--output', type=str, help='path to save dataset')
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-chat-hf", help="path to load tokenizer")
    parser.add_argument('--env', type=str, default="gsm8k", help="env name")
    parser.add_argument('--temperature', type=float, default=0.4, help="temperature parameter used in reward mapping")
    parser.add_argument('--only_success', action='store_true', help="only keep success trajectory when activate")
    parser.add_argument('--criteria', choices=['all', 'success', 'best', 'better'], default="all", help="all: loss on all actions; success: loss on only the success actions; best: loss on the best actions among a trajectory; better: loss on actions that are better than history")
    parser.add_argument('--remove_duplication', action='store_true', help="remove duplicates")
    parser.add_argument('--shuffle', action='store_true', help="shuffle the dataset")


    args = parser.parse_args()
    env = args.env

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    with open(args.filepath) as f:
        logs = json.load(f)
        dataset = []
        for key in logs:
            log = logs[key]
            if skip_dialogue(args.only_success, log):
                continue
            dataset += convert_to_data(log, args)
    data_analysis(dataset)
    if args.shuffle:
        random.shuffle(dataset)
    
    weighted_dataset = exp_dataset_reward(deep_copy_dataset(dataset), temperature=args.temperature)
    save_dataset(weighted_dataset, args.output)
    
if __name__ == "__main__":
    main()
