import json
import openai
import asyncio

# Load settings from the JSON file
with open('./Settings.json') as f:
    settings = json.load(f)
api_key = settings["OpenAi_API_Key"]

# Initialize the OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

model = "gpt-4o"

def create_completion(data):
    return client.chat.completions.create(**data)

def create_completion(data):
    return client.chat.completions.create(**data)

async def LLM_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None
        
async def Input_Expansion_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None

async def Inner_Monologue_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None

async def Intuition_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None

async def Final_Response_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None

async def Short_Term_Memory_API_Call(API, backend_model, conversation, username, bot_name):
    try:
        data = {
            "model": model,
            "messages": conversation
        }

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(None, create_completion, data)
        
        assistant_message = completion.choices[0].message.content.strip()
        return assistant_message
    except Exception as e:
        print(e)
        return None