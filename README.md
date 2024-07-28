# ChatBot

## Introduction 
In this project, we will develop an AI assistant to assist customers at a food store. The **RAG code** has been modified from [this repo](https://github.com/pixegami/langchain-rag-tutorial) and the remaining code was entirely written by **Khôi** and **Khang**.

## Install dependencies

1. Create an anaconda environment.
```python
conda create --name [environment-name] python=3.10.14
```

2. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


3. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

4. We are going to use Llama 3 available on Ollama. Install Ollama running this one-liner. Please refer [here](https://github.com/ollama/ollama?tab=readme-ov-file) for more information.

```python
curl -fsSL https://ollama.com/install.sh | sh
```

Then run the following command to pull llama 3.1-8b-instruct model.
```python
ollama pull llama3.1
```

## Create database
1. Put the name of your database and the data path in the .env file.
```python
CHROMA_PATH = "food_database"
DATA_PATH = "shop_data"
```

2. Several example data located at `shop_data`. You can add your custom data.

Create the Chroma DB.

```python
export PYTHONPATH=$(pwd)
python backend/create_database.py
```

## Run chatbot app

```python
python app.py
```

**Please note that** the response time may vary depending on the resources available on your computer (12 GB VRAM at least).