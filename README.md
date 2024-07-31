# ChatBot

## Introduction 
In this project, we will develop an AI assistant to assist customers at a food store. The **RAG code** has been modified from [this repo](https://github.com/pixegami/langchain-rag-tutorial) and the remaining code was entirely written by **Khôi** and **Khang**.

## Install dependencies

1. Create an anaconda environment.
```python
conda create --name [environment-name] python==3.12.*
```

2. Setting up the Environment on Ubuntu
To set up the environment, Ubuntu users can simply execute the following command:
```python
./setup.sh
``` 
Please note that this project utilizes Llama 3, which is available on Ollama.

## Create database
1. Put the name of your database and the data path in the .env file.
```python
CHROMA_PATH = "data/chromadb"
DATA_PATH = "data/shop_data"
```

2. Several example datasets are located in the `shop_data` directory. You can also add your custom data as needed.

Chroma DB is automatically created when you set up the environment using the `setup.sh` script. If you need to create a new Chroma DB, you can do so by running the following command:
```python
task db
```

## Run chatbot app

```python
task run
```

**Please note that** the response time may vary depending on the resources available on your computer (12 GB VRAM at least).