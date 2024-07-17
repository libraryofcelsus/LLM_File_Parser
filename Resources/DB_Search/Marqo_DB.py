import json
import requests
import sys
import time
from datetime import datetime
from uuid import uuid4
from marqo import Client as MarqoClient
from marqo.errors import MarqoWebError, BackendCommunicationError
import traceback
import importlib
import re

Debug_Output = False

def timestamp_func():
    try:
        return time.time()
    except:
        return time()

def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str

# Check if local Marqo server is running
def check_local_server_running():
    try:
        response = requests.get("http://localhost:8882")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Initialize Marqo client
def initialize_client():
    with open('./Settings.json', 'r') as file:
        settings = json.load(file)
    marqo_url = settings.get('Marqo_URL', 'http://localhost:8882')
    if check_local_server_running():
        marqo_url = 'http://localhost:8882'
    else:
        marqo_url = settings.get('Marqo_URL', 'http://localhost:8882')
    client = MarqoClient(url=marqo_url)
    return client

mq = initialize_client()

def search_db(collection_name, bot_name, user_id, expanded_input, selected_domain):
    timestamp = timestamp_func()
    timestring = timestamp_to_datetime(timestamp)

    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    vector_db = settings.get('Vector_DB', 'Marqo_DB')

    try:
        db_search_module_name = f'Resources.DB_Search.{vector_db}'
        db_search_module = importlib.import_module(db_search_module_name)

        client = db_search_module.initialize_client()
    except:
        traceback.print_exc()
        return []  # Return empty list if initialization fails
        
    def remove_duplicate_dicts(input_list):
        if input_list is None:
            return []
        output_list = []
        for item in input_list:
            if item not in output_list:
                output_list.append(item)
        return output_list
        
    all_db_search_results = []

    try:
        response = client.index(f"BOT_NAME_{bot_name}").search(
            q=expanded_input,
            limit=18,
            filter_string=f"user:{user_id}",
            search_method="TENSOR"
        )
        vector_results = response.get('hits', [])
        all_db_search_results.extend(vector_results)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")

    domains = [domain.strip() for domain in re.split(r',\s*', selected_domain)]      
    for domain in domains:
        try:
            response = client.index(f"BOT_NAME_{bot_name}_Domains").search(
                q=expanded_input,
                limit=3,
                filter_string=f"user:{user_id}",
                search_method="TENSOR"
            )
            hits = response.get('hits', [])
            for hit in hits:
                domain = hit['domain']
                try:
                    response = client.index(f"BOT_NAME_{bot_name}").search(
                        q=expanded_input,
                        limit=5,
                        filter_string=f"user:{user_id}",
                        search_method="TENSOR"
                    )
                    vector_results = response.get('hits', [])
                    all_db_search_results.extend(vector_results)
                except Exception as e:
                    print(f"\nAn unexpected error occurred: {str(e)}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")

    all_db_search_results = remove_duplicate_dicts(all_db_search_results)
    
    if all_db_search_results:
        try:
            sorted_results = sorted(all_db_search_results, key=lambda x: x['_score'])

            top_30_results = sorted_results[-30:]
            sorted_results = [f"{hit['message']}" for hit in top_30_results]
            return sorted_results
        except Exception as e:
            print(f"\nAn unexpected error occurred while sorting: {str(e)}")

    return []
    
    
    

def retrieve_domain_list(collection_name, bot_name, user_id, expanded_input):
    timestamp = timestamp_func()
    timestring = timestamp_to_datetime(timestamp)
    collection_name = f"{collection_name}_Domains"

    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return []  # Return empty list if settings fail to load

    vector_db = settings.get('Vector_DB', 'Qdrant_DB')

    try:
        db_search_module_name = f'Resources.DB_Search.{vector_db}'
        db_search_module = importlib.import_module(db_search_module_name)
        
        client = db_search_module.initialize_client()
    except Exception as e:
        traceback.print_exc()
        return []  # Return empty list if initialization fails

    try:
        hits = client.index(collection_name).search(
            q=expanded_input,
            limit=25,
            filter_string=f"user:{user_id}",
            search_method="TENSOR"
        )
        domain_list = [hit['domain'] for hit in hits.get('hits', [])]
    except Exception as e:
        if "Not found: Collection" in str(e):
            print(f"Collection not found: {collection_name}")
        else:
            print(f"An unexpected error occurred: {str(e)}")
        domain_list = []  # Return empty list if there's an error

    return domain_list
