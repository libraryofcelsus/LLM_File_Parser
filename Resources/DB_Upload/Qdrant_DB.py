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

def timestamp_func():
    try:
        return time.time()
    except:
        return time()
        
def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str

# Singleton for SentenceTransformer
class SentenceTransformerSingleton:
    _model = None

    @staticmethod
    def get_model():
        if SentenceTransformerSingleton._model is None:
            SentenceTransformerSingleton._model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return SentenceTransformerSingleton._model

def embeddings(model, query):
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

client = initialize_client()
model = SentenceTransformerSingleton.get_model()

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

def upload_document(collection_name, bot_name, user_id, text, domain, filename, semantic_db_term):
    timestamp = timestamp_func()
    timestring = timestamp_to_datetime(timestamp)
    vector = embeddings(model, semantic_db_term + '\n' + text)
    unique_id = str(uuid4())
    point_id = unique_id + str(int(timestamp))

    metadata = {
        'bot': bot_name,
        'time': timestamp,
        'message': text,
        'timestring': timestring,
        'uuid': unique_id,
        'user': user_id,
        'source': filename,
        'tag': semantic_db_term,
        'domain': domain.upper(),
        'memory_type': 'External_Resources'
    }
    
    # Ensure the collection is initialized before uploading
    embed_size = len(vector)
    initialize_collection(client, collection_name, embed_size)
    
    client.upsert(collection_name=collection_name,
                  points=[PointStruct(id=unique_id, vector=vector, payload=metadata)])  


