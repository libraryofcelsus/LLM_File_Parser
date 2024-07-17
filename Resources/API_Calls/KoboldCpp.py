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
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 1500
        }

        timeout = aiohttp.ClientTimeout(total=60)  # Increase the timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        
async def Input_Expansion_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 100
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        
async def Domain_Selection_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 100
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        

async def Inner_Monologue_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 500
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        
        
async def Intuition_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 550
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        
        
async def Final_Response_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 1000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        
        
async def Short_Term_Memory_API_Call(API, backend_model, prompt, username, bot_name):
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        HOST = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001')
        url = f"{HOST}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "mode": "instruct",
            "instruction_template": backend_model,
            "messages": prompt,
            "max_tokens": 400
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, ssl=False) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        assistant_message = response_json['choices'][0]['message']['content']
                        return assistant_message
                    except ValueError:
                        print("Response content is not valid JSON:", await response.text())
                        return None
                else:
                    print("Failed to get a valid response:", response.status)
                    return None

    except Exception as e:
        traceback.print_exc()
        return None
        