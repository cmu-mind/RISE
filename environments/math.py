import gym
import numpy as np
import pandas as pd
# import nltk  # Add this import at the beginning of your script
import re
import json
import os
from environments.math_equivalence import is_equiv


MESSAGES = {
    "SUCCESS": "Great job! You've successfully answered the question."
}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
class MATHEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, data_folder):
        super(MATHEnv, self).__init__()
        self.name = "MATH"
        self.data = read_jsonl(data_folder)
    def reset(self, idx):
        self.current_step = idx
        self.current_data = self.data[self.current_step]

        self.query = self.current_data['problem']
        self.answer = self.current_data['solution']
        return self.get_observation()

    def get_observation(self, reward=None, error_message=None):
        if reward == None or error_message == None:
            observation = f"Great job! You've successfully answered the question. Now, let's move on to the next one.\n{self.query}"
        else:
            observation = f"The answer is incorrect. Please try again. Here's the question: {self.query}"
        return observation

    def format_output(self, error_message, success, reward):
        if success:
            return MESSAGES["SUCCESS"]
        else:
            return self.get_observation(reward, error_message)

    def step(self, action):
        return self.evaluate_response(action)


    def extract_answer(self, input_string):
        pattern = r'boxed\{((?:[^{}]|(?:{[^}]*}))*)\}'
        matches = re.findall(pattern, input_string)
        if matches:
            return matches[-1]
        else:
            return None
    

    def evaluate_response(self, assistant_response):
        assistant_answer = self.extract_answer(assistant_response)
        answer = self.extract_answer(self.answer)
        
        if is_equiv(assistant_answer, answer):
            return 1, MESSAGES["SUCCESS"], True
        else:
            return 0, "The answer is incorrect. Please try again.", False