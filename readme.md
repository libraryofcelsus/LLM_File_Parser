# LLM File Parser
Version 0.02 of LLM File Parser by [LibraryofCelsus.com](https://www.libraryofcelsus.com)  
  
[Installation Guide](#installation-guide)  
[Skip to Changelog](#changelog)  
[Discord Server](https://discord.gg/pb5zcNa7zE)

------
**Recent Changes**

• 07/17 Added Knowledge Domains as seperate index so existing domains can be searched with vectors.

• 07/12 First Release

------

### What is this project?

This project is part of my larger Aetherius project and is designed to streamline the process of transforming unstructured data into structured databases and datasets. The program falls under AutoML and uses various LLM techniques to scan, chunk, and summarize unstructured documents, transforming them into structured data with minimal user input.  

Current supported file types: .epub, .pdf, .txt, .png, .jpg, .jpeg, .mp4, .mkv, .flv, and .av  

Chatbots using this format:   
- https://github.com/libraryofcelsus/Hierarchical_RAG_Chatbot

This project serves as the document uploader for: https://github.com/libraryofcelsus/Advanced_RAG_Chatbot

Main Ai Assistant Github: https://github.com/libraryofcelsus/Aetherius_AI_Assistant  

------

My Ai work is self-funded by my day job, consider supporting me if you appreciate my work.

<a href='https://ko-fi.com/libraryofcelsus' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

------

Join the Discord for help or to get more in-depth information!

Discord Server: https://discord.gg/pb5zcNa7zE

Subscribe to my youtube for Video Tutorials: https://www.youtube.com/@LibraryofCelsus (Channel not Launched Yet)

Code Tutorials available at: https://www.libraryofcelsus.com/research/public/code-tutorials/

Made by: https://github.com/libraryofcelsus


------


# Changelog: 
**0.02** 

• Added Knowledge Domains as seperate index so existing domains can be searched with vectors. 

**0.01** 

• First Release


# Installation Guide

## Installer bat

Download the project zip folder by pressing the <> Code drop down menu.

**1.** Install Python 3.10.6, Make sure you add it to PATH: **https://www.python.org/downloads/release/python-3106/**

**2.** Run "install_requirements.bat" to install the needed dependencies.  The bat will install Git, Poppler, Tesseract, FFmpeg, and the needed python dependencies.  

(If you get an error when installing requirements run: **python -m pip cache purge**)

**3.** Set up Qdrant or Marqo DB.  To change what DB is used, edit the "Vector_DB" Key in ./Settings.json.  Qdrant is the default. 

Qdrant Docs: https://qdrant.tech/documentation/guides/installation/   

Marqo Docs: https://docs.marqo.ai/2.9/  

To use a local Qdrant server, first install Docker: https://www.docker.com.  
Next type: **docker pull qdrant/qdrant:v1.9.1** in the command prompt.  
After it is finished downloading, type **docker run -p 6333:6333 qdrant/qdrant:v1.9.1**  

To use a local Marqo server, first install Docker: https://www.docker.com.  
Next type: **docker pull marqoai/marqo:latest** in the command prompt.  
After it is finished downloading, type **docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest**   

(If it gives an error, check the docker containers tab for a new container and press the start button.  Sometimes it fails to start.)  

See: https://docs.docker.com/desktop/backup-and-restore/ for how to make backups.  

Once the local Vector DB server is running, it should be auto detected by the scripts.   

**6.** Install your desired API.  (Not needed if using OpenAi)  To change what Api is used, edit the "API" Key in ./Settings.json  
https://github.com/oobabooga/text-generation-webui  
https://github.com/LostRuins/koboldcpp  

**8.** Launch a script with one of the **run_*.bat**  

**9.** Change the information inside of the "Settings" tab to your preferences.  

**10.** Put a file in its corresponding folder in the ./Uploads directory.  The Uploads folder will be created when first running the File Processing Script.  

To get Whisper working with cuda, you may need to run the commands:    
**.\venv\Scripts\activate**   
**pip uninstall torch torchaudio**   
**pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu118/torch_stable.html**  


If you wish to change the format in which data is uploaded to the Vector DB, the upload scripts can be found in ./Resources/DB_Upload  

-----

## About Me

In January 2023, I had my inaugural experience with ChatGPT and LLMs in general. Since that moment, I've been deeply obsessed with AI, dedicating countless hours each day to studying it and to hands-on experimentation.

# Contact
Discord: libraryofcelsus      -> Old Username Style: Celsus#0262

MEGA Chat: https://mega.nz/C!pmNmEIZQ