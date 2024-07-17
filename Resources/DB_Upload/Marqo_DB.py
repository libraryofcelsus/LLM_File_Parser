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

Debug_Output = True  # Enable debug output

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

def upload_document(collection_name, bot_name, user_id, text, domain, filename, semantic_db_term):
    try:
        timestamp = timestamp_func()
        timestring = timestamp_to_datetime(timestamp)

        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        vector_db = settings.get('Vector_DB', 'Marqo_DB')
        try:
            db_search_module_name = f'Resources.DB_Search.{vector_db}'
            db_search_module = importlib.import_module(db_search_module_name)

            client = db_search_module.initialize_client()
        except Exception as e:
            traceback.print_exc()
            return []  # Return empty list if initialization fails

        # Check if the index exists
        try:
            mq.index(collection_name).get_stats()
        except MarqoWebError as e:
            if 'index_not_found' in str(e):
                try:
                    mq.create_index(collection_name)
                except Exception as create_error:
                    print(f"Error creating index: {str(create_error)}")
                    return False
            else:
                raise e

        unique_id = str(uuid4())
        point_id = unique_id + str(int(timestamp))

        # Create the document with all metadata
        document = {
            "_id": point_id,
            "bot": bot_name,
            "time": timestamp,
            "message": text,
            "timestring": timestring,
            "uuid": unique_id,
            "user": user_id,
            "domain": domain.upper(),
            "memory_type": "External_Resources",
            "document": text  # The main content to be embedded
        }

        # Add the document to the Marqo index
        mq.index(collection_name).add_documents(
            [document],
            tensor_fields=["document"],  # Field to be embedded
            client_batch_size=1  # Process one document at a time
        )

        # Check if the index exists
        try:
            mq.index(f"{collection_name}_Domains").get_stats()
        except MarqoWebError as e:
            if 'index_not_found' in str(e):
                try:
                    mq.create_index(f"{collection_name}_Domains")
                except Exception as create_error:
                    print(f"Error creating index: {str(create_error)}")
                    return False
            else:
                raise e

        domain_search = set()
        try:
            hits = client.index(f"{collection_name}_Domains").search(
                q=domain.upper(),
                limit=25,
                filter_string=f"user:{user_id}",
                search_method="TENSOR"
            )
            if hits and 'hits' in hits:
                for hit in hits['hits']:
                    domain_hit = hit.get('domain', None)
                    if domain_hit:
                        domain_search.add(domain_hit.upper())

            # If the domain was not found in the search results, add it
            if domain.upper() not in domain_search:
                print(f"Domain {domain.upper()} not found, adding to index")
                unique_id = str(uuid4())
                point_id = unique_id + str(int(timestamp))

                # Create the document with all metadata
                document = {
                    "_id": point_id,
                    "bot": bot_name,
                    "message": text,
                    "user": user_id,
                    "domain": domain.upper(),
                    "document": domain.upper()  # The main content to be embedded
                }

                # Add the document to the Marqo index
                mq.index(f"{collection_name}_Domains").add_documents(
                    [document],
                    tensor_fields=["document"],  # Field to be embedded
                    client_batch_size=1  # Process one document at a time
                )

        except Exception as e:
            print(f"\nAn unexpected error occurred during domain search: {str(e)}")
            hits = "No Collection"

        if Debug_Output:
            print(f"Successfully added document with ID: {point_id}")
        return True

    except BackendCommunicationError as e:
        print(f"Error uploading document: {str(e)}")
        return False
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        return False