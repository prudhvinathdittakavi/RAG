import numpy as np
import llama_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import math
import os
from dotenv import load_dotenv
import PyPDF2
import logging

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# import utils
from src.utils import chunk_text, get_embedding_model, get_embeddings, cosine_similarity,  read_file_based_on_type

# Configuring the logger
# different logging levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Creating a custom logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
palm_api_key = os.getenv('PALM_API_KEY', "")

# Configure the API key
genai.configure(api_key=palm_api_key)

# Read the external text file and split it into chunks.
file_path = input("Enter the path to the file which you wanted to ask your questions: ")

# Read the file based on the file type
text_file = read_file_based_on_type(file_path)

# parameters to tune
chunk_size_range = [250, 500]
top_k_range = [5, 10]


def tune_document_rag(document_path: str, training_file_path: str, chunk_size_range: list, top_k_range: list):
    """
    Function to tune the parameters of the RAG model for a given document.

    Args:
        document_path (str): The path to the document file.
        training_file_path (str): The path to the training file.
        chunk_size_range (list): A list of integers representing the range of chunk sizes to be tested.
        top_k_range (list): A list of integers representing the range of top_k values to be tested.

    Returns:
        dict: A dictionary containing the best chunk size and top_k values for the RAG model.
    """
    # Read the text file
    text = read_text_file(document_path)

    # generate tuples of (chunk_size, top_k)
    pairs_to_train = [(chunk_size, top_k) for chunk_size in chunk_size_range for top_k in top_k_range]

    for chunk_size, top_k in pairs_to_train:
        # Chunk the text into chunks of a specified size
        chunks = chunk_text(text, chunk_size)

        # Get the embedding model


    


