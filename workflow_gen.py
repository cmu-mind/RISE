import json
import os
import re
import torch
import argparse, json, os, re
from tqdm import tqdm
from typing import Dict, List
import sys
from rich import print

from policies import ChatGPTPolicy, FastChatPolicy

from environments.gsm8k import GSM8KEnv
from environments.math import MATHEnv

parser = argparse.ArgumentParser(description='N-turn evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, default="/Users/yxqu/Desktop/CMU/Research/Self-Imrovement/data/gsm8k/demo.jsonl", help='path to dataset to evaluate on')
parser.add_argument('--dialogue_limit', type=int, default=20, help='maximum number of turns in the policy\'s dialogue to keep')
parser.add_argument('--env', choices=['gsm8k', 'math'], default="gsm8k", help='environment to run eval on')
parser.add_argument('--log_dir', type=str, default="./", help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, default=5, help='max number of interaction turns')
parser.add_argument('--models', nargs='+', type=str, help="models to use for policy")
parser.add_argument('--controller_address', type=str, default="21002", help="model to use for policy")
parser.add_argument('--num_of_samples', nargs='+', type=int, help='number of actions generated each turn')
parser.add_argument('--verbose', action='store_true', help="print out logs")

args = parser.parse_args()
print(args)

def detect_duplicates(text, threshold):
    # Convert the text to lowercase and split it into words
    words = text.lower().split()

    # Create a dictionary to store word counts
    word_counts = {}

    # Count the occurrences of each word
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Check if any word count exceeds the threshold
    for word, count in word_counts.items():
        if count > threshold:
            return True
        if len(word) > 300:
            return True
    else:
        return False


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
        
class ExperimentWrapper():
    def __init__(self, args):
        self.args = args

        # Set environment (No logging for env)
        self.env = None
        if args.env == 'gsm8k':
            self.env = GSM8KEnv(args.data_path)
        elif args.env == 'math':
            self.env = MATHEnv(args.data_path)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{args.max_turns}_turns.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}

    def init_policy(self, model):
        if "gpt" in model:
            self.policy = ChatGPTPolicy(dialogue_limit=args.dialogue_limit, model=model)
            self.role = "assistant"
        else:
            self.policy = FastChatPolicy(dialogue_limit=args.dialogue_limit, model=model, controller_address=args.controller_address)
            self.role = "agent"

    def construct_policy_dialogue(self, observations, actions, model):
        self.init_policy(model)
        self.policy.reset(self.args.env)
        for i in range(len(actions)):
            action = actions[i]
            observation = observations[i]
            self.policy.dialogue.append({"role": "user", "content": observation})
            self.policy.dialogue.append({"role": self.role, "content": action})
        self.policy.dialogue.append({"role": "user", "content": observations[-1]})

    def run_expr(self):
        try:
            for idx in tqdm(range(0,len(self.env.data)), disable=self.args.verbose):
                observation, reward, valid_action = None, None, None
                turn_history = {"best_actions": [], "actions": {}, "best_observations": [], "observations": {}, "best_rewards": [], "rewards": {}}
                
                init_observation = self.env.reset(idx)
                query = self.env.query
                if self.args.verbose:
                    print(f'------\nQuery {idx}: {query}')    

                for turn in range(self.args.max_turns):
                    self.construct_policy_dialogue([init_observation] + turn_history["best_observations"], turn_history["best_actions"], args.models[turn])
                    actions = []
                    try:
                        while len(actions) == 0:
                            actions = self.policy.forward(args.num_of_samples[turn])
                            for action in actions:
                                if detect_duplicates(action, 150):
                                    print(f"[WARNING] Index {idx}: Duplicate detected in action: {action}")
                                    actions.remove(action)
                                
                    except (ValueError, TypeError) as e:
                        print(f"[ERROR] Index {idx}: {e}")
                        # Logging
                        turn_history["actions"][turn] = ["blocked"]
                        turn_history["rewards"][turn] = [None]
                        break

                    turn_history["actions"][turn] = actions
                    turn_history["rewards"][turn] = []
                    turn_history["observations"][turn] = []

                    for action in actions:
                        reward, error_message, success = self.env.step(action)
                        observation = self.env.format_output(error_message, success, reward)
                        turn_history["rewards"][turn].append(reward)
                        turn_history["observations"][turn].append(observation)
                    
                    max_reward_idx = turn_history["rewards"][turn].index(max(turn_history["rewards"][turn]))
                    
                    best_action = turn_history["actions"][turn][max_reward_idx]
                    best_reward = turn_history["rewards"][turn][max_reward_idx]
                    best_observation = turn_history["observations"][turn][max_reward_idx]

                    turn_history["best_actions"].append(best_action)
                    turn_history["best_rewards"].append(best_reward)
                    turn_history["best_observations"].append(best_observation)

                    if self.args.verbose:
                        print(f"- Turn {turn}")
                        print(f"-- Best Action: {best_action}")
                        print(f"-- Best Observation: {best_observation}")
                        print(f"-- Best Reward: {best_reward}")

                    if turn_history["best_rewards"][-1] == 1.0:
                        break

                max_reward = max(turn_history["best_rewards"])
                answer = self.env.answer
                log_episode = {
                    "task_id": idx,
                    "query": query,
                    "answer": answer,
                    "max_reward": max_reward,
                    "turn_history": turn_history,
                    "dialogue": self.policy.dialogue
                }
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn+1}")
            print(f"log path: {self.log_path}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()

if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()



