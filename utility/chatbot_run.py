import os
import argparse

from typing import List

from bot import ChatBot

def read_des(desc_path: str = "data/desc_summ.txt"):
    enabled, information = None, None
    if os.path.exists(desc_path):
        with open(desc_path, "r") as f:
            desc_summ = f.readlines()

        # Remove desc pạth
        os.remove(desc_path)

        enabled = desc_summ[0].strip("\n")
        information = ''.join(desc_summ[1:])

    return enabled, information

def initialize_chatbot():
    # Initialize chatbot, input prompt and get response
    model = ChatBot(model_id="llama3.1")
    
    return model

    # return formatted_response