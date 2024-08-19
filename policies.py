import re

from typing import Tuple

from utils import CompletionGPT, ChatGPT, DIALOGUES

import requests
import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import List, Dict, Any, Union

# import TimeoutException
from requests.exceptions import Timeout, ConnectionError
from fastchat.model.model_adapter import get_conversation_template

import copy

class Agent:
    def __init__(self, **configs) -> None:
        self.name = configs.pop("name", None)
        self.src = configs.pop("src", None)
        pass

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError

class Prompter:
    @staticmethod
    def get_prompter(prompter_name: Union[str, None]):
        # check if prompter_name is a method and its variable
        if not prompter_name:
            return None
        if hasattr(Prompter, prompter_name) and callable(getattr(Prompter, prompter_name)):
            return getattr(Prompter, prompter_name)
    
    @staticmethod
    def claude(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "Human",
            "agent": "Assistant",
        }
        for item in messages:
            prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
        prompt += "Assistant:"
        return {"prompt": prompt}

    @staticmethod
    def openchat_v3_1(messages: List[Dict[str, str]]):
        prompt = "Assistant is GPT4<|end_of_turn|>"
        role_dict = {
            "user": "User: {content}<|end_of_turn|>",
            "agent": "Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "Assistant:"
        return {"prompt": prompt}
    
    @staticmethod
    def openchat_v3_2(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "GPT4 User: {content}<|end_of_turn|>",
            "agent": "GPT4 Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "GPT4 Assistant:"
        return {"prompt": prompt}

class FastChatAgent(Agent):
    def __init__(self, model_name, controller_address=None, worker_address=None, temperature=0, max_new_tokens=32, top_p=0, prompter=None, args=None, **kwargs) -> None:
        if controller_address is None and worker_address is None:
            raise ValueError("Either controller_address or worker_address must be specified.")
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            worker_addr = controller_addr
        if worker_addr == "":
            return
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args
        }
        if self.prompter:
            prompt = self.prompter(history)
            gen_params.update(prompt)
        else:
            conv = get_conversation_template(self.model_name)

            for history_item in history:
                role = history_item["role"]
                content = history_item["content"]
                if role == "user" or role == "system":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            if history[-1]["role"] == conv.roles[1]:
                print("finish agent's response")
            else:
                print("generate a new response")
                conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            gen_params.update({
                "prompt": prompt,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
            })
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        text = json.loads(line)["text"]
                return text
            # if timeout or connection error, retry
            except Timeout: 
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(60)
        else:
            raise Exception("Timeout after 3 retries.")

class BasePolicy:
    def __init__(self):
        pass

    def forward(query, observation, available_actions):
        raise NotImplementedError
    
    def init_dialogue(self, role, env):
        if env in DIALOGUES:
            dialogue = copy.deepcopy(DIALOGUES[env])
            if len(dialogue) >= 2:
                dialogue[1]["role"] = role
        else:
            dialogue = []
        return dialogue

class ChatGPTPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "gpt-4-turbo-preview", response_limit: int = 1000):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit
        print(f"Teacher Model is {self.model}")

    def reset(self, env):
        self.dialogue = self.init_dialogue("assistant", env)

    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        actions = []

        raw_actions = ChatGPT(self.dialogue, model=self.model, num_samples=num_of_samples)
        # for action in raw_actions:
        #     if action not in actions:
        #         actions.append(action)
        return raw_actions


class FastChatPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "", response_limit: int = 1000, controller_address=21002):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit

        self.agent = FastChatAgent(model_name=model, controller_address=f"http://localhost:{controller_address}", worker_address=None, temperature=1.0, max_new_tokens=response_limit, top_p=1.0, prompter=None, args=None, name="FastChatAgent")
        
    def reset(self, env):
        self.dialogue = self.init_dialogue("agent", env)

    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        actions = []
        for i in range(num_of_samples):
            raw_actions = self.agent.inference(self.dialogue)
            action = raw_actions[0] if isinstance(raw_actions, list) else raw_actions
            # if action not in actions:
            actions.append(action)
        return actions