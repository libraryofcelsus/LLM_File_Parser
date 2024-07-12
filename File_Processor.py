import os
import json
import time
from datetime import datetime
import shutil
import importlib
import traceback
import aiofiles
import asyncio
import pytesseract
from PIL import Image
from ebooklib import epub
from bs4 import BeautifulSoup
from queue import Queue
import base64
import re
import gc
import subprocess
from pdf2image import convert_from_path, pdfinfo_from_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTImage, LTTextBox, LTTextLine, LTFigure, LTTextBoxHorizontal
from pdfminer.high_level import extract_pages
from pdfminer.converter import PDFPageAggregator
import openai
from io import BytesIO
import cv2
import numpy as np
from functools import partial
import whisper
import requests

TOOLS_DIR = os.path.join(os.path.dirname(__file__), 'Tools')
POPPLER_PATH = os.path.join(TOOLS_DIR, 'poppler', 'Library', 'bin')
TESSERACT_PATH = os.path.join(TOOLS_DIR, 'tesseract')

pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

Debug_Output = True
Dataset_Output = False

async def create_upload_folders():
    subfolders = [
        './Uploads/TXT',
        './Uploads/TXT/Finished',
        './Uploads/PDF',
        './Uploads/PDF/Finished',
        './Uploads/EPUB',
        './Uploads/EPUB/Finished',
        './Uploads/VIDEOS',
        './Uploads/VIDEOS/Finished',
        './Uploads/SCANS/Finished',
        './Uploads/LOGS',
        './Datasets'
    ]

    async def create_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            if Debug_Output:
                print(f"Created folder: {folder}")

    await asyncio.gather(*(create_folder(folder) for folder in subfolders))

async def open_file(filepath):
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as file:
        return await file.read().strip()

def timestamp_func():
    try:
        return time.time()
    except:
        return time()

def is_url(string):
    return string.startswith('http://') or string.startswith('https://')

def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str

def import_api_function():
    settings_path = './Settings.json'
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    api_module_name = settings['API']
    module_path = f'./Resources/API_Calls/{api_module_name}.py'
    spec = importlib.util.spec_from_file_location(api_module_name, module_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    llm_api_call = getattr(api_module, 'LLM_API_Call', None)
    input_expansion_api_call = getattr(api_module, 'Input_Expansion_API_Call', None)
    inner_monologue_api_call = getattr(api_module, 'Inner_Monologue_API_Call', None)
    intuition_api_call = getattr(api_module, 'Intuition_API_Call', None)
    final_response_api_call = getattr(api_module, 'Final_Response_API_Call', None)
    short_term_memory_response_api_call = getattr(api_module, 'Short_Term_Memory_Response_API_Call', None)
    if llm_api_call is None:
        raise ImportError(f"LLM_API_Call function not found in {api_module_name}.py")
    return llm_api_call, input_expansion_api_call, inner_monologue_api_call, intuition_api_call, final_response_api_call, short_term_memory_response_api_call

def find_base64_encoded_json(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    pattern = re.compile(b'[A-Za-z0-9+/]{100,}={0,2}')
    matches = pattern.findall(binary_data)
    valid_json_objects = []
    for match in matches:
        try:
            decoded_data = base64.b64decode(match)
            json_data = json.loads(decoded_data)
            valid_json_objects.append(json_data)
        except (base64.binascii.Error, json.JSONDecodeError):
            continue
    return valid_json_objects

def load_format_settings(backend_model):
    file_path = f'./Model_Formats/{backend_model}.json'
    if (os.path.exists(file_path)):
        with open(file_path, 'r') as file:
            formats = json.load(file)
    else:
        formats = {
            "user_input_start": "", 
            "user_input_end": "", 
            "assistant_input_start": "", 
            "assistant_input_end": "",
            "system_input_start": "", 
            "system_input_end": ""
        }
    return formats

def set_format_variables(backend_model):
    format_settings = load_format_settings(backend_model)
    heuristic_input_start = format_settings.get("heuristic_input_start", "")
    heuristic_input_end = format_settings.get("heuristic_input_end", "")
    system_input_start = format_settings.get("system_input_start", "")
    system_input_end = format_settings.get("system_input_end", "")
    user_input_start = format_settings.get("user_input_start", "")
    user_input_end = format_settings.get("user_input_end", "")
    assistant_input_start = format_settings.get("assistant_input_start", "")
    assistant_input_end = format_settings.get("assistant_input_end", "")
    return heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end

def format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, response):
    try:
        if response is None:
            return "ERROR WITH API"
        if backend_model == "Llama_3":
            assistant_input_start = "assistant"
            assistant_input_end = "assistant"
        botname_check = f"{botnameupper}:"
        while (response.startswith(assistant_input_start) or response.startswith('\n') or
               response.startswith(' ') or response.startswith(botname_check)):
            if response.startswith(assistant_input_start):
                response = response[len(assistant_input_start):]
            elif response.startswith(botname_check):
                response = response[len(botnameupper):]
            elif response.startswith('\n'):
                response = response[1:]
            elif response.startswith(' '):
                response = response[1:]
            response = response.strip()
        botname_check = f"{botnameupper}: "
        if response.startswith(botname_check):
            response = response[len(botname_check):].strip()
        if backend_model == "Llama_3":
            if "assistant\n" in response:
                index = response.find("assistant\n")
                response = response[:index]
        if response.endswith(assistant_input_end):
            response = response[:-len(assistant_input_end)].strip()
        return response
    except:
        traceback.print_exc()
        return ""


def write_dataset_simple(filename, user_input, output):
    data = {
        "input": user_input,
        "output": output
    }
    try:
        with open(f'./Datasets/{filename}_dataset.json', 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(f'./Datasets/{filename}_dataset.json', 'w') as file:
            json.dump([data], file, indent=4)


async def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
    

async def Text_Extract(client, db_module):
    await create_upload_folders()
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    bot_name = settings.get('Bot_Name', '')
    username = settings.get('Username', '')
    user_id = settings.get('User_ID', '')
    print("Enter a knowledge domain to assign to the files. To have the LLM assign one per chunk, type: 'Auto'")
    Domain = input().strip()
    await create_upload_folders()
    
    semaphore = asyncio.Semaphore(8)  # Limit concurrent tasks for better performance
    host_queue = asyncio.Queue()
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        host_data = settings.get('HOST_AetherNode', '').strip()
        hosts = host_data.split(' ')
    except Exception as e:
        print(f"An error occurred while reading the host file: {e}")
    
    for host in hosts:
        host_queue.put_nowait(host)
    
    while True:
        try:
            timestamp = time.time()
            timestring = timestamp_to_datetime(timestamp)
            await process_files_in_directory('./Uploads/SCANS', './Uploads/SCANS/Finished', Domain, semaphore, host_queue, client, db_module, bot_name, username)
            await process_files_in_directory('./Uploads/TXT', './Uploads/TXT/Finished', Domain, semaphore, host_queue, client, db_module, bot_name, username)
            await process_files_in_directory('./Uploads/PDF', './Uploads/PDF/Finished', Domain, semaphore, host_queue, client, db_module, bot_name, username)
            await process_files_in_directory('./Uploads/EPUB', './Uploads/EPUB/Finished', Domain, semaphore, host_queue, client, db_module, bot_name, username)
            await process_files_in_directory('./Uploads/VIDEOS', './Uploads/VIDEOS/Finished', Domain, semaphore, host_queue, client, db_module, bot_name, username)
            gc.collect()
        except:
            traceback.print_exc()

async def process_files_in_directory(directory_path, finished_directory_path, Domain, semaphore, host_queue, client, db_module, bot_name, username, chunk_size=700, overlap=80):
    try:
        files = os.listdir(directory_path)
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        
        total_files = len(files)
        if total_files == 0:
            return  # No files to process

    #    print_progress_bar(0, total_files, prefix=f'Processing {directory_path}:', suffix='Complete', length=50)

        async with semaphore:
            tasks = [process_and_move_file(directory_path, finished_directory_path, file, Domain, semaphore, host_queue, client, db_module, bot_name, username, chunk_size, overlap) for file in files]
            
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                await task
                print_progress_bar(i, total_files, prefix=f'Processing {directory_path}:', suffix='Complete', length=50)
        
        gc.collect()
    except Exception as e:
        print(e)
        traceback.print_exc()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def detect_sections(text):
    sections = {}
    sections['references'] = [m.start() for m in re.finditer(r'\bReferences\b', text, re.IGNORECASE)]
    sections['titles'] = [m.start() for m in re.finditer(r'\bChapter \d+\b', text, re.IGNORECASE)]
    sections['abstracts'] = [m.start() for m in re.finditer(r'\bAbstract\b', text, re.IGNORECASE)]
    return sections

def extract_useful_text(text):
    sections = detect_sections(text)
    useful_text = text

    if 'references' in sections and sections['references']:
        useful_text = useful_text[:sections['references'][0]]

    return useful_text.strip()

# Ensure that OpenCV uses GPU if available
cv2.setUseOptimized(True)

async def convert_pdf_pages(pdf_path, poppler_path, start, end, dpi=200):
    """
    Convert a range of pages from the PDF to images.
    """
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        pages = await loop.run_in_executor(executor, partial(convert_from_path, pdf_path, dpi=dpi, first_page=start, last_page=end, poppler_path=poppler_path))
    return pages

async def analyze_page(executor, loop, page, index, text_density_threshold):
    try:
        black_pixel_density = await loop.run_in_executor(executor, partial(process_page, page))
        if black_pixel_density > text_density_threshold:
            return index, page  # Return the page along with the index
    except Exception as e:
        print(f"Error processing page {index}: {e}")
    return None, None

async def detect_images_in_pdf(pdf_path, poppler_path=POPPLER_PATH, text_density_threshold=0.12, batch_size=10):
    """
    Detects images in a PDF by analyzing the density of black pixels after converting pages to binary images.
    Returns a list of pages that likely contain images.
    """
    image_pages = []
    image_objects = []  # To store the image objects for further processing

    # Get the total number of pages
    info = await asyncio.to_thread(pdfinfo_from_path, pdf_path, poppler_path=poppler_path)
    total_pages = info["Pages"]

    print_progress_bar(0, total_pages, prefix='Checking PDF for Images:', suffix='Complete', length=50)

    completed_tasks = 0

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        pages = await convert_pdf_pages(pdf_path, poppler_path, start, end)
        
        tasks = [analyze_page(ProcessPoolExecutor(), asyncio.get_running_loop(), pages[i - start], i, text_density_threshold) for i in range(start, end + 1)]

        for task in asyncio.as_completed(tasks):
            result, page = await task
            if result is not None:
                image_pages.append(result)
                image_objects.append(page)
            completed_tasks += 1
            print_progress_bar(completed_tasks, total_pages, prefix='Checking PDF for Images:', suffix='Complete', length=50)

    return image_pages, image_objects

def process_page(page_image):
    """
    Processes a single PDF page image to determine if it contains a significant amount of non-text content.
    Returns the black pixel density of the page.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using histogram equalization
    enhanced_image = cv2.equalizeHist(gray_image)
    
    # Convert to binary image
    _, binary_image = cv2.threshold(enhanced_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Calculate black pixel density
    black_pixel_density = np.sum(binary_image == 0) / binary_image.size
    
    return black_pixel_density

def extract_image_from_pdf_page(pdf_path, page_number):
    """
    Extracts images from a specific page of a PDF and returns them as PIL Image objects.
    """
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path=POPPLER_PATH)
    return images[0]  # Assuming one image per page for simplicity

def detect_images_in_epub(epub_path):
    image_chapters = []
    book = epub.read_epub(epub_path)
    
    for item in book.get_items_of_type(9):
        soup = BeautifulSoup(item.content, 'html.parser')
        if soup.find('img'):
            image_chapters.append(item.get_name())
    
    return image_chapters

async def gpt_vision(query, image, filename, Domain, db_module):
    """
    Processes the image using GPT-4 Vision to generate a detailed description.
    """
    try:
        # Read the API key from Settings.json
        with open('./Settings.json', 'r') as file:
            settings = json.load(file)
        bot_name = settings.get('Bot_Name', '')
        botnameupper = bot_name.upper()
        username = settings.get('Username', '')
        user_id = settings.get('User_ID', '')
        API = settings.get('API', 'AetherNode')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        LLM_Model = settings.get('LLM_Model', 'AetherNode')  
        LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_Response_API_Call = import_api_function()
        heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
        api_key = settings.get("OpenAi_API_Key")
        if not api_key:
            return "Error: OpenAi_API_Key not found in Settings.json"
        collection_name = f"BOT_NAME_{bot_name}"
        # Encode the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Create the prompt
        prompt = """You are an AI sub-module for visual-to-text conversion in documents. Your tasks:

1. Analyze scanned documents.
2. Identify all images and data visualizations (charts, graphs, tables).
3. Convert each visual element into a detailed text description in simple paragraph format, including:
   - For images: describe the subject, colors, shapes, any text within the image, and the overall impression or context.
   - For data visuals: describe the type, labels, key data points, and main insights.
4. Ignore supplemental text that is not associated with the visual elements.
5. Ensure the descriptions are clear and concise."""
        sub_prompt = ""
        if len(query) > 2:
            sub_prompt = f"\nPlease consider the user's inquiry in your response, tailoring the data gathered to it.\nUSER INQUIRY: {query}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt} {sub_prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        # Make the API call
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()

        # Extract and print the actual response content
        if 'choices' in response_data and len(response_data['choices']) > 0:
            response_text = response_data['choices'][0]['message']['content']
            
            semanticterm = list()
            semanticterm.append({'role': 'system', 'content': "MAIN SYSTEM PROMPT: You are a bot responsible for tagging articles with a question-based title for database queries. Your task is to read the provided text and generate a concise title in the form of a question that accurately represents the article's content. The title should be semantically identical to the article's overview, without including any extraneous information. Use the format: [<QUESTION TITLE>]."})

            semanticterm.append({'role': 'assistant', 'content': f"GIVEN ARTICLE: {response_text}"})

            semanticterm.append({'role': 'user', 'content': "Create a brief, single question that encapsulates the semantic meaning of the article. Use the format: [<QUESTION TITLE>]. Please only provide the question title, as it will be directly appended to the article."})

            semanticterm.append({'role': 'assistant', 'content': "ASSISTANT: Sure! Here's the semantic question tag for the article: "})

            text_to_remove = f"ASSISTANT: Sure! Here's the semantic question tag for the article: "

            prompt = ''.join([message_dict['content'] for message_dict in semanticterm])

            semantic_db_term = await Final_Response_API_Call(API, backend_model, semanticterm, username, bot_name)
            semantic_db_term = re.sub(r'[^\w\s\.,!?;:]', '', semantic_db_term)
            if semantic_db_term.startswith(text_to_remove):
                semantic_db_term = semantic_db_term[len(text_to_remove):].strip()
            if 'cannot provide a summary of' in semantic_db_term.lower():
                semantic_db_term = 'Tag Censored by Model'
            semanticterm.clear()
            
            if Domain == "Auto":
                domain_extraction = []
                domain_extraction = [
                    {'role': 'system', 'content': "You are a knowledge domain extractor. Your task is to identify the single, most general knowledge domain that best represents the given text. Respond with only one word for the domain, without any explanation or specifics."},
                    {'role': 'user', 'content': f"Text to analyze: {semantic_db_term} - {response_text}"},
                    {'role': 'assistant', 'content': "The most relevant knowledge domain for the given text is: "}
                ]
                text_to_remove = f"DOMAIN EXTRACTOR: The most relevant knowledge domain for the given text is: "
                text_to_remove2 = f"DOMAIN EXTRACTOR:"
                extracted_domain = await Final_Response_API_Call(API, backend_model, domain_extraction, username, bot_name)
                extracted_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, extracted_domain)
                extracted_domain = re.sub(r'[^\w\s]', '', extracted_domain)
                if extracted_domain.startswith(text_to_remove):
                    extracted_domain = extracted_domain[len(text_to_remove):].strip()
                if extracted_domain.startswith(text_to_remove2):
                    extracted_domain = extracted_domain[len(text_to_remove2)].strip()
                Domain = extracted_domain
                domain_extraction.clear()
            
                print('\n---------')
                print(f"{filename}")
                print(f"\nGENERATED INPUT: {semantic_db_term}")
                print(f"EXTRACTED DOMAIN: {Domain}")
                print(f"\nOUTPUT: {response_text}")
            db_module.upload_document(
                collection_name, bot_name, user_id, response_text, Domain.upper(), filename, semantic_db_term
            )
            return response_text
        else:
            print("Full API response:", response_data)
            return "No valid response from the API."
    except Exception as e:
        print(f"Error in gpt_vision: {e}")
        return "Error processing image."

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

async def process_and_move_file(directory_path, finished_directory_path, file, Domain, semaphore, host_queue, client, db_module, bot_name, username, chunk_size, overlap):
    async with semaphore:
        try:
            file_path = os.path.join(directory_path, file)
            text = None

            # Determine file type and process accordingly
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.pdf':
                # Detect images in PDF concurrently
                image_pages, image_objects = await detect_images_in_pdf(file_path)
                if image_pages:
                    print(f"PDF file {file} contains images on pages: {image_pages}")
                    image_descriptions = await asyncio.gather(*[gpt_vision("", image, file, Domain, db_module) for image in image_objects])
                    # Handle the image descriptions here, e.g., save to a file or database
                
                # Extract text from PDF concurrently
                text = await extract_text_from_pdf_with_ocr(file_path)

            elif file_extension == '.epub':
                # Detect images in EPUB
                image_chapters = await asyncio.to_thread(detect_images_in_epub, file_path)
                if image_chapters:
                    print(f"ePub file {file} contains images in chapters: {image_chapters}")

                # Extract text from EPUB concurrently
                text = await chunk_text_from_file(file_path, Domain, host_queue, client, db_module, chunk_size, overlap)

            elif file_extension in ['.mp4', '.mkv', '.flv', '.avi']:
                # Extract text from video files concurrently
                text = await chunk_text_from_file(file_path, Domain, host_queue, client, db_module, chunk_size, overlap)

            else:
                # Extract text from other file types concurrently
                text = await chunk_text_from_file(file_path, Domain, host_queue, client, db_module, chunk_size, overlap)

            if text:
                # Process the extracted text
                cleaned_text = clean_text(text)
                useful_text = extract_useful_text(cleaned_text)
                
                # Chunk the text concurrently
                chunks = await parallel_chunk_text(useful_text, chunk_size, overlap)
                total_chunks = len(chunks)
                collection_name = f"BOT_NAME_{bot_name}"

                # Initialize the tasks list and ensure proper sequencing
                tasks = []
                for chunk in chunks:
                    host = await host_queue.get()
                    tasks.append(asyncio.create_task(summarize_chunk_and_release_host(host, chunk, collection_name, bot_name, username, client, file_path, Domain, db_module, host_queue)))

                for i, task in enumerate(asyncio.as_completed(tasks), 1):
                    await task
                    print_progress_bar(i, total_chunks, prefix='Summarizing:', suffix='Complete', length=50)

            # Move the processed file to the finished directory
            finished_file_path = os.path.join(finished_directory_path, file)
            await asyncio.to_thread(shutil.move, file_path, finished_file_path)
        except Exception as e:
            print(e)
            traceback.print_exc()

async def summarize_chunk_and_release_host(host, chunk, collection_name, bot_name, username, client, file_path, Domain, db_module, host_queue):
    try:
        await summarized_chunk_from_file(host, chunk, collection_name, bot_name, username, client, file_path, Domain, db_module)
    finally:
        await host_queue.put(host)


async def chunk_text_from_file(file_path, Domain, host_queue, client, db_module, chunk_size=600, overlap=50):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    bot_name = settings.get('Bot_Name', '')
    username = settings.get('Username', '')
    user_id = settings.get('User_ID', '')
    API = settings.get('API', 'AetherNode')
    Web_Search = settings.get('Search_Web', 'False')
    backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
    LLM_Model = settings.get('LLM_Model', 'AetherNode')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_Response_API_Call = import_api_function()
    try:
        if Debug_Output:
            print("Reading given file, please wait...")
        texttemp = None
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.txt':
            async with aiofiles.open(file_path, 'r') as file:
                texttemp = (await file.read()).replace('\n', ' ').replace('\r', '')
        elif file_extension == '.pdf':
            texttemp = await extract_text_from_pdf_with_ocr(file_path)
        elif file_extension == '.epub':
            book = epub.read_epub(file_path)
            texts = []
            for item in book.get_items_of_type(9):
                soup = BeautifulSoup(item.content, 'html.parser')
                texts.append(soup.get_text())
            texttemp = ' '.join(texts)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image = Image.open(file_path)
            if image is not None:
                texttemp = pytesseract.image_to_string(image).replace('\n', ' ').replace('\r', '')
        elif file_extension in ['.mp4', '.mkv', '.flv', '.avi']:
            audio_file = "audio_extracted.wav"
            subprocess.run(["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", "-f", "wav", audio_file])

            model_stt = whisper.load_model("tiny")
            transcribe_result = model_stt.transcribe(audio_file)
            if isinstance(transcribe_result, dict) and 'text' in transcribe_result:
                texttemp = transcribe_result['text']
            else:
                print("Unexpected transcribe result")
                texttemp = ""  
            os.remove(audio_file)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

        texttemp = '\n'.join(line for line in texttemp.splitlines() if line.strip())
        return texttemp
    except Exception as e:
        print(e)
        traceback.print_exc()
        return "Error"

async def parallel_chunk_text(text, chunk_size, overlap):
    chunks = []
    total_length = len(text)

    def chunk_task(start):
        end = min(start + chunk_size, total_length)
        return text[start:end]

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, chunk_task, start)
            for start in range(0, total_length, chunk_size - overlap)
        ]

        total_tasks = len(tasks)
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            chunks.append(await task)
            print_progress_bar(i, total_tasks, prefix='Chunking Text:', suffix='Complete', length=50)

    return chunks

async def extract_text_from_pdf_with_ocr(file_path):
    text = ""
    info = await asyncio.to_thread(pdfinfo_from_path, file_path, poppler_path=POPPLER_PATH)
    total_pages = info["Pages"]
    print(f'Total pages to process: {total_pages}')
    
    def convert_page(page_num):
        return convert_from_path(file_path, dpi=200, first_page=page_num, last_page=page_num, poppler_path=POPPLER_PATH)[0]

    tasks = [asyncio.to_thread(convert_page, i) for i in range(1, total_pages + 1)]
    
    images = []
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        image = await task
        images.append(image)
        print_progress_bar(i, total_pages, prefix='Converting PDF:', suffix='Complete', length=50)
    
    total_images = len(images)
    ocr_tasks = [asyncio.to_thread(pytesseract.image_to_string, image) for image in images]
    
    ocr_results = []
    for i, task in enumerate(asyncio.as_completed(ocr_tasks), 1):
        result = await task
        ocr_results.append(result)
        print_progress_bar(i, total_images, prefix='Gathering OCR Data:', suffix='Complete', length=50)
    
    text = ''.join(ocr_results)
    return text

async def summarized_chunk_from_file(host, chunk, collection_name, bot_name, username, client, file_path, Domain, db_module):
    try:
        botnameupper = bot_name.upper()
        filelist = [] 
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        API = settings.get('API', 'AetherNode')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        LLM_API_Call, Input_Expansion_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
        filesum = list()
        filesum.append({
            'role': 'system',
            'content': """
            You are an AI assistant specializing in condensing articles while retaining all key information. Your task is to read the provided text and continue the conversation by presenting a concise version that:

            1. Keeps all essential information, including names, dates, numbers, and specific details.
            2. Uses natural language to present the information in a conversational manner.
            3. Maintains the original meaning and context.
            4. Shortens the text where possible without losing any content.
            5. Does not add any new information or interpretations.

            If no article is provided, simply respond with "I don't have any article to work with. What would you like to discuss?"

            Remember: Your goal is to present all the information from the original text in a more concise, conversational way.
            """
        })
        filesum.append({'role': 'user', 'content': f"Here's an article I'd like you to condense: {chunk}"})
        filesum.append({'role': 'assistant', 'content': "I've read the article you provided. I will now provide a concise version that retains all the key information.\nHere is a condensed version of the text: "})
        text_to_remove = f"SUMMARIZED ARTICLE: Based on the scraped text, here is the summary: "
        heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
        user_id = settings.get('User_ID', 'USER_ID')
        conv_length = settings.get('Conversation_Length', '3')
        Web_Search = settings.get('Search_Web', 'False')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        LLM_Model = settings.get('LLM_Model', 'AetherNode')

        Write_Dataset = settings.get('Write_To_Dataset', 'False')
        Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
        Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
        prompt = ''.join([message_dict['content'] for message_dict in filesum])
        text = await Final_Response_API_Call(API, backend_model, filesum, username, bot_name)
        if text.startswith(text_to_remove):
            text = text[len(text_to_remove):].strip()
        if len(text) < 20:
            text = "No File available."

        fileyescheck = 'yes'
        if 'no file' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no article' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no summary' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'provide the article' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'i am an ai' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no article provided' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        elif 'no file available' in text.lower():
            print('---------')
            print('Summarization Failed')
            return
        else:
            if 'cannot provide a summary of' in text.lower():
                text = chunk
            if 'yes' in fileyescheck.lower():
                semanticterm = list()
                semanticterm.append({'role': 'system', 'content': "MAIN SYSTEM PROMPT: You are a bot responsible for tagging articles with a question-based title for database queries. Your task is to read the provided text and generate a concise title in the form of a question that accurately represents the article's content. The title should be semantically identical to the article's overview, without including any extraneous information. Use the format: [<QUESTION TITLE>]."})

                semanticterm.append({'role': 'assistant', 'content': f"GIVEN ARTICLE: {text}"})

                semanticterm.append({'role': 'user', 'content': "Create a brief, single question that encapsulates the semantic meaning of the article. Use the format: [<QUESTION TITLE>]. Please only provide the question title, as it will be directly appended to the article."})

                semanticterm.append({'role': 'assistant', 'content': "ASSISTANT: Sure! Here's the semantic question tag for the article: "})

                text_to_remove = f"ASSISTANT: Sure! Here's the semantic question tag for the article: "

                prompt = ''.join([message_dict['content'] for message_dict in semanticterm])

                semantic_db_term = await Final_Response_API_Call(API, backend_model, semanticterm, username, bot_name)
                semantic_db_term = re.sub(r'[^\w\s\.,!?;:]', '', semantic_db_term)
                if semantic_db_term.startswith(text_to_remove):
                    semantic_db_term = semantic_db_term[len(text_to_remove):].strip()
                filename = os.path.basename(file_path)
                if 'cannot provide a summary of' in semantic_db_term.lower():
                    semantic_db_term = 'Tag Censored by Model'
                filelist.append(filename + ' ' + text)

                base_name = os.path.splitext(filename)[0]
                logs_dir = './Uploads/LOGS'
                text_file_path = os.path.join(logs_dir, base_name + '.txt')
                with open(text_file_path, 'a', encoding='utf-8') as f:
                    f.write('<' + filename + '>\n')
                    f.write('<' + semantic_db_term + '>\n')
                    f.write('<' + text + '>\n\n')

                if Domain == "Auto":
                    domain_extraction = []
                    domain_extraction = [
                        {'role': 'system', 'content': "You are a knowledge domain extractor. Your task is to identify the single, most general knowledge domain that best represents the given text. Respond with only one word for the domain, without any explanation or specifics."},
                        {'role': 'user', 'content': f"Text to analyze: {semantic_db_term} - {text}"},
                        {'role': 'assistant', 'content': "The most relevant knowledge domain for the given text is: "}
                    ]
                    text_to_remove = f"DOMAIN EXTRACTOR: The most relevant knowledge domain for the given text is: "
                    text_to_remove2 = f"DOMAIN EXTRACTOR:"
                    extracted_domain = await Final_Response_API_Call(API, backend_model, domain_extraction, username, bot_name)
                    extracted_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, extracted_domain)
                    extracted_domain = re.sub(r'[^\w\s]', '', extracted_domain)
                    if extracted_domain.startswith(text_to_remove):
                        extracted_domain = extracted_domain[len(text_to_remove):].strip()
                    if extracted_domain.startswith(text_to_remove2):
                        extracted_domain = extracted_domain[len(text_to_remove2)].strip()
                    Domain = extracted_domain
                print('\n---------')
                print(f"{filename}")
                print(f"\nGENERATED INPUT: {semantic_db_term}")
                print(f"EXTRACTED DOMAIN: {extracted_domain}")
                print(f"\nOUTPUT: {text}")
                db_module.upload_document(
                    collection_name, bot_name, user_id, text, Domain.upper(), filename, semantic_db_term
                )
    
                dataset = []
                filename_base = os.path.splitext(filename)[0]
                heuristic_input_start2, heuristic_input_end2, system_input_start2, system_input_end2, user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2 = set_format_variables(Dataset_Format)

                Dataset_System_Prompt = settings.get('Dataset_System_Prompt')
                Dataset_System_Prompt = Dataset_System_Prompt.replace('<<botname>>', bot_name)
                Dataset_System_Prompt = Dataset_System_Prompt.replace('<<username>>', username)

                dataset.append({'role': 'system', 'content': f"{system_input_start2}{Dataset_System_Prompt}{system_input_end2}"})
                dataset.append({'role': 'user', 'content': f"{user_input_start2}{semantic_db_term}{user_input_end2}"})

                filtered_content = [entry['content'] for entry in dataset if entry['role'] in ['system', 'user', 'assistant']]
                llm_input = '\n'.join(filtered_content)

                assistant_response = f"{assistant_input_start2}{text}{assistant_input_end2}"
                if Dataset_Output:
                    print(f"\n\nINPUT: {llm_input}")  
                    print(f"\n\nRESPONSE: {assistant_response}")

                if Write_Dataset == 'True':
                    write_dataset_simple(filename_base, llm_input, assistant_response)
                    print(f"Written to ./Datasets/{filename_base}_dataset.json\n\n")
                
                
                pass
            else:
                print('---------')
                print(f'\n\n\nERROR MESSAGE FROM BOT: {fileyescheck}\n\n\n')
        table = filelist
        return table
    except Exception as e:
        print(e)
        traceback.print_exc()
        table = "Error"
        return table
        
        

async def main():
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    vector_db = settings.get('Vector_DB', 'Qdrant_DB')
    
    db_module_name = f'Resources.DB_Upload.{vector_db}'
    db_module = importlib.import_module(db_module_name)
    
    client = db_module.initialize_client()
    username = settings.get('Username', 'User')
    user_id = settings.get('User_ID', 'UNIQUE_USER_ID')
    bot_name = settings.get('Bot_Name', 'Chatbot')
    history = []
    while True:
        response = await Text_Extract(client, db_module)

if __name__ == "__main__":
    asyncio.run(main())
