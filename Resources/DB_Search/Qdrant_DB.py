import json
import requests
import sys
import time
from datetime import datetime
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, Filter, FieldCondition, 
                                  Range, MatchValue)
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import traceback
import importlib
import re

def timestamp_func():
    try:
        return time.time()
    except:
        return time()
        
def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embed_size = "768"

def embeddings(query):
    vector = model.encode([query])[0].tolist()
    return vector


def check_local_server_running():
    try:
        response = requests.get("http://localhost:6333/dashboard/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Initialize Qdrant client
def initialize_client():
    client = None
    if check_local_server_running():
        client = QdrantClient(url="http://localhost:6333")
    else:
        try:
            with open('./Settings.json', 'r') as file:
                settings = json.load(file)
            api_key = settings.get('Qdrant_API_Key', '')
            client = QdrantClient(url="https://qdrant-api-url.com", api_key=api_key)
        except:
            print("\n\nQdrant is not started. Please enter API Keys or run Qdrant Locally.")
            sys.exit()
    return client



def initialize_collection(client, collection_name, embed_size):
    try:
        collection_info = client.get_collection(collection_name=collection_name)
    except:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embed_size, distance=Distance.COSINE),
            )
        except:
            traceback.print_exc()

def search_db(collection_name, bot_name, user_id, expanded_input, selected_domain):
    timestamp = timestamp_func()
    timestring = timestamp_to_datetime(timestamp)
    
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    vector_db = settings.get('Vector_DB', 'Qdrant_DB')
    
    def remove_duplicate_dicts(input_list):
        if input_list is None:
            return []
        output_list = []
        for item in input_list:
            if item not in output_list:
                output_list.append(item)
        return output_list
        
    try:
        db_search_module_name = f'Resources.DB_Search.{vector_db}'
        db_search_module = importlib.import_module(db_search_module_name)
        
        client = db_search_module.initialize_client()
    except:
        traceback.print_exc()
        return []  # Return empty list if initialization fails

    all_db_search_results = []
    vector = embeddings(expanded_input)
    
    try:
        hits = client.search(
            collection_name=f"BOT_NAME_{bot_name}",
            query_vector=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user",
                        match=MatchValue(value=f"{user_id}")
                    )
                ]
            ),
            limit=15
        )
        all_db_search_results.extend(hits)
    except Exception as e:
        if "Not found: Collection" in str(e):
            search1 = "No Collection"
        else:
            print(f"\nAn unexpected error occurred: {str(e)}")
            search1 = "No Collection"
            
            
            
            
    domains = [domain.strip() for domain in re.split(r',\s*', selected_domain)]
            
            
            
            
    for domain in domains:
        vector1 = embeddings(domain)
        try:
            hits = client.search(
                collection_name=f"BOT_NAME_{bot_name}_DOMAINS",
                query_vector=vector1,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user",
                            match=MatchValue(value=f"{user_id}")
                        )
                    ]
                ),
                limit=3
            )
            for hit in hits:
                domain = hit['domain']
                try:
                    hits = client.search(
                        collection_name=f"BOT_NAME_{bot_name}",
                        query_vector=vector,
                        query_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="domain",
                                    match=MatchValue(value=domain),
                                ),
                                FieldCondition(
                                    key="user",
                                    match=MatchValue(value=f"{user_id}"),
                                ),
                            ]
                        ),
                        limit=5
                    )
                    all_db_search_results.extend(hits)
                except Exception as e:
                    if "Not found: Collection" in str(e):
                        search2 = "No Collection"
                    else:
                        print(f"\nAn unexpected error occurred: {str(e)}")
                        search2 = "No Collection"
        except Exception as e:
            if "Not found: Collection" in str(e):
                search1 = "No Collection"
            else:
                print(f"\nAn unexpected error occurred: {str(e)}")
                search1 = "No Collection"

    all_db_search_results = remove_duplicate_dicts(all_db_search_results)
    
    if all_db_search_results:
        sorted_results = sorted(all_db_search_results, key=lambda hit: hit.score, reverse=True)
        top_30_results = sorted_results[-30:]
        sorted_results = [f"{hit.payload['message']}" for hit in top_30_results]
        return sorted_results

        
    return []


def retrieve_domain_list(collection_name, bot_name, user_id, expanded_input):
    timestamp = timestamp_func()
    timestring = timestamp_to_datetime(timestamp)
    collection_name = f"{collection_name}_Domains"
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    vector_db = settings.get('Vector_DB', 'Qdrant_DB')

    try:
        db_search_module_name = f'Resources.DB_Search.{vector_db}'
        db_search_module = importlib.import_module(db_search_module_name)
        
        client = db_search_module.initialize_client()
    except:
        traceback.print_exc()
        return []  # Return empty list if initialization fails
        
    vector = embeddings(expanded_input)
    
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user",
                        match=MatchValue(value=f"{user_id}")
                    )
                ]
            ),
            limit=25
        )

        domain_list = [hit.payload['domain'] for hit in hits]
    except Exception as e:
        if "Not found: Collection" in str(e):
            domain_list = "No Collection"
            print(f"\nAn unexpected error occurred: {str(e)}")
        else:
            print(f"\nAn unexpected error occurred: {str(e)}")
            domain_list = "No Collection"
            
    
    return domain_list
