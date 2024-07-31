#!/bin/bash

# Check if 'task' command is available
if ! command -v task &> /dev/null; then
    echo "'task' is not installed. Installing..."
    sudo snap install task --classic

    # Verify installation
    if command -v task &> /dev/null; then
        echo "'task' has been successfully installed."
    else
        echo "Failed to install 'task'. Please check for errors."
    fi
else
    echo "'task' is already installed."
fi


# # create enviroment conda 
# conda create -n chatbot python==3.12.* -y
# source activate base 
# conda activate chatbot

echo "Run Task setup"
task

echo "Waiting for 3 seconds before running ollama pull..."
sleep 3

# Run ollama pull command
ollama pull llama3.1


# load create database
task db