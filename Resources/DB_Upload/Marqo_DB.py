import json
import requests
import sys
import time
from datetime import datetime
from uuid import uuid4
from marqo import Client as MarqoClient
from marqo.errors import MarqoWebError, BackendCommunicationError

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

def upload_document(collection_name, bot_name, user_id, text, domain, filename, semantic_db_term):
    try:
        # Check if the index exists
        try:
            mq.index(collection_name).get_stats()
        except MarqoWebError as e:
            if e.code == 'index_not_found':
                try:
                    mq.create_index(collection_name)
                except Exception as create_error:
                    print(f"Error creating index: {str(create_error)}")
                    return False
            else:
                raise e

        timestamp = timestamp_func()
        timestring = timestamp_to_datetime(timestamp)
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

        if Debug_Output:
            print(f"Successfully added document with ID: {point_id}")
        return True

    except BackendCommunicationError as e:
        print(f"Error uploading document: {str(e)}")
        return False
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        return False
