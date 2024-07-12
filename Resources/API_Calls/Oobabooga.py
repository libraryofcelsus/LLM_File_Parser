import sys
import os
import json
import time
import datetime as dt
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import numpy as np
import re
import keyboard
import traceback
import asyncio
import aiofiles
import aiohttp



async def LLM_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "temperature": 0.6,
                "top_p": 0.8,
                "repetition_penalty": 1.3,
                "top_k": 30,
                "max_tokens": 700,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]                   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()
    
async def Input_Expansion_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "temperature": 0.6,
                "top_p": 0.8,
                "repetition_penalty": 1.20,
                "top_k": 40,
                "max_tokens": 100,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()
    
        
async def Inner_Monologue_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()
        
        
async def Intuition_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()
        
        
async def Final_Response_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "temperature": 0.6,
                "top_p": 0.8,
                "repetition_penalty": 1.25,
                "top_k": 40,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()
        
        
async def Short_Term_Memory_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        url = "http://127.0.0.1:5000/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        history = prompt

        while True:
       #     history.append({"role": "user", "content": user_message})
            data = {
                "mode": "instruct",
                "instruction_template": backend_model,
        #        "character": "Assistant",
                "messages": history,
                "stop_sequence": ["###", "### ", "```", "\n\n"],
                "stopping_strings": ["###", "### ", "```", "\n\n"],
                "stop": ["###", "### ", "```", "\n\n"]   
            }

            response = requests.post(url, headers=headers, json=data, verify=False)
            assistant_message = response.json()['choices'][0]['message']['content']
     #       history.append({"role": "assistant", "content": assistant_message})
      #      print(assistant_message)
            return assistant_message
    except:
        traceback.print_exc()