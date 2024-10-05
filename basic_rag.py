import numpy as np
import llama_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as palm
from llama_index.llms.palm import PaLM
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

palm_api_key = os.getenv('PALM_API_KEY', "") 
palm.configure(api_key=palm_api_key)

def read_text_file(file_path: str):
    """
    Function to read the contents of a text file
    - file_path: The path to the file as a string
    - Returns: The content of the file as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text(text, chunk_size):
    """
    Function to break a large text into smaller chunks of a specified size
    - text: The text string to be chunked
    - chunk_size: The size of each chunk (number of characters)
    - Returns: A list of text chunks (each chunk is a string)  
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def get_embedding_model():
    """
    Function to initialize and return a pre-trained embedding model
    - Uses the HuggingFaceEmbedding class to load a specific model
    - Returns: An embedding model object that can generate embeddings for text
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model

def get_embeddings(embed_model, text: str):
    """
    Function to get the text embeddings for a given text using a provided embedding model
    - embed_model: The pre-trained embedding model
    - text: The input text string for which embeddings are required
    - Returns: The embedding vector as a list of floats
    """
    embeddings = embed_model.get_text_embedding(text)
    return embeddings

def dot_product(vec1, vec2):
    """
    Function to compute the dot product of two vectors
    - vec1: The first vector (list of floats)
    - vec2: The second vector (list of floats)
    - Returns: The dot product of the two vectors (a float) 
    """
    return sum(a * b for a, b in zip(vec1, vec2))

def magnitude(vec):
    """
    Function to calculate the magnitude (Euclidean norm) of a vector
    - vec: The input vector (list of floats)
    - Returns: The magnitude of the vector (a float)  
    """
    return math.sqrt(sum(v**2 for v in vec))

def cosine_similarity(vec1, vec2):
    """
    Function to compute the cosine similarity between two vectors
    - vec1: The first vector (list of floats)
    - vec2: The second vector (list of floats)
    - Returns: The cosine similarity score (a float between -1 and 1)
    """
    dot_prod = dot_product(vec1, vec2)
    mag_vec1 = magnitude(vec1)
    mag_vec2 = magnitude(vec2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0  # Handle division by zero

    return dot_prod / (mag_vec1 * mag_vec2)

# Read the external text file and split it into chunks.
text_file = read_text_file("data/J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt")
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
q_embd = get_embeddings(embd_model, query)

# Generate a similarity score betweent QE and each of the Chunk embeddings.
ratings = [cosine_similarity(q_embd, x) for x in vdb]

# Extract top K chunks based on similarity score.
k = 5
idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted

# Frame a prompt with the query and the top-k chunks
prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
    Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[chunks[index] for index in idx]}"

# Prompt an LLM with the prompt.
model = PaLM(api_key=palm_api_key)
output = model.complete(prompt)
print(output.text)