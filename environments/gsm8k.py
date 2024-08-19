import gym
import numpy as np
import pandas as pd
# import nltk  # Add this import at the beginning of your script
import re
import json

MESSAGES = {
    "SUCCESS": "Great job! You've successfully answered the question."
}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
class GSM8KEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, data_file):
        super(GSM8KEnv, self).__init__()
        self.name = "GSM8K"
        self.data = read_jsonl(data_file)
    
    def reset(self, idx):
        self.current_step = idx
        self.current_data = self.data[self.current_step]
        self.query = self.current_data['question']
        self.answer = self.current_data['answer']
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


    def extract_answers_from_llm(self, text):
        pattern = r'boxed{([^}]+)}'
        matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        last_match = matches[-1]
        content = last_match.replace(",", "")
        
        try:
            answer = int(content)
            return answer
        except:
            return None
    
        # pattern = r'boxed\{(\d+)\}'
        # matches = re.findall(pattern, input_string)
        # if matches:
        #     return matches[-1]
        # else:
        #     return None
    
    def extract_answers(self, input_string):
        # Regular expression pattern to match step_answers and answer
        step_answers_pattern = r'<<(.*?)>>'
        answer_pattern = r'####\s*([-+]?\d+(?:\.\d+)?)'

        # Find the step_answers in the input string
        step_answers_matches = re.findall(step_answers_pattern, input_string)
        step_answers = [eval(match.split('=')[-1]) for match in step_answers_matches]

        # Find the answer in the input string
        answer_match = re.search(answer_pattern, input_string)
        answer = answer_match.group(1) if answer_match else None

        return step_answers, answer
    

    def evaluate_response(self, assistant_response):

        assistant_answer = self.extract_answers_from_llm(assistant_response)
        step_answers, answer = self.extract_answers(self.current_data['answer'])
        
        if assistant_answer is None:
            return -1, "The answer is incorrect. Please try again.", False
        elif assistant_answer is not None and float(assistant_answer) == float(answer):
            return 1, MESSAGES["SUCCESS"], True
        else:
            return 0, "The answer is incorrect. Please try again.", False