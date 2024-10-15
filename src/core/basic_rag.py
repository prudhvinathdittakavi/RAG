import numpy as np
import llama_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import math
import os
from dotenv import load_dotenv
import PyPDF2
import logging
import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# import utils
from src.utils.utils import chunk_text, read_text_file, get_embedding_model, get_embeddings, cosine_similarity

# Configuring the logger
# different logging levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Creating a custom logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

palm_api_key = os.getenv('PALM_API_KEY', "")
genai.configure(api_key=palm_api_key)

# Read the external text file and split it into chunks.
text_file = read_text_file("../../data/J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt")
chunks = chunk_text(text_file, chunk_size=250)

# Initialize an embedding Model.
embd_model = get_embedding_model()

# Generate Embedding for each chunk.
vdb = [] # our sasta vector database :-)
for chunk in chunks:
    embd = get_embeddings(embd_model, chunk)
    vdb.append(embd)

# Generate Embedding for Query.
query = "what is greatest fear of Dursleys"
logging.info(f"Query: {query}")
q_embd = get_embeddings(embd_model, query)

# Generate a similarity score betweent QE and each of the Chunk embeddings.
ratings = [cosine_similarity(q_embd, x) for x in vdb]

# Extract top K chunks based on similarity score.
k = 5
idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted

# Frame a prompt with the query and the top-k chunks
prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
    Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[chunks[index] for index in idx]}"

# Log the prompt (for debugging or tracking purposes)
logging.debug(f"Prompt: {prompt}")

# Prompt an LLM with the prompt.
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
logging.info(f"Response: {response.text}")