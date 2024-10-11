import numpy as np
import llama_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import math
import os
from dotenv import load_dotenv
import PyPDF2
import logging

# import utils
from src.utils import chunk_text, extract_text_from_pdf, get_embedding_model, get_embeddings, cosine_similarity

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

# read the pdf file and extract the text
pdf_path = "data/nasdaq-nvda-2023-10K-23668751.pdf"
text_file = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text_file, chunk_size=250)

# Initialize an embedding Model.
embd_model = get_embedding_model()

# Generate Embedding for each chunk.
vdb = [] # our sasta vector database :-)
for chunk in chunks:
    embd = get_embeddings(embd_model, chunk)
    vdb.append(embd)

while True:
    # Get the user's query
    query = input("Ask your question (or type 'exit' to quit): ")
    
    # Exit condition
    if query.lower() == 'exit':
        print("Goodbye!")
        break

    logging.info(f"Query: {query}")

    # Generate embedding for the query
    q_embd = get_embeddings(embd_model, query)

    # Generate similarity scores between the query embedding and each chunk embedding
    ratings = [cosine_similarity(q_embd, x) for x in vdb]

    # Extract top K chunks based on similarity score
    k = 5
    idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted

    # Frame a prompt with the query and the top-k chunks
    relevant_info = [chunks[index] for index in idx]
    prompt = f"""
    You are a smart agent. A question would be asked to you and relevant information would be provided.
    Your task is to answer the question using the information provided.
    Question: {query}.
    Relevant Information: {relevant_info}.
    """
    
    # Log the prompt (for debugging or tracking purposes)
    logging.debug(f"Prompt: {prompt}")

    # Generate a response using the generative model
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    # # Print the response in the terminal
    # print(f"Answer: {response.text}")
    
    # Log the response (optional)
    logging.info(f"Answer: {response.text}")


# # Generate Embedding for Query.
# query = "What is the total revenue of the company this year?"
# logging.info(f"Query: {query}")
# q_embd = get_embeddings(embd_model, query)

# # Generate a similarity score betweent QE and each of the Chunk embeddings.
# ratings = [cosine_similarity(q_embd, x) for x in vdb]

# # Extract top K chunks based on similarity score.
# k = 5
# idx = np.argpartition(ratings, -k)[-k:]  # Indices not sorted

# # Frame a prompt with the query and the top-k chunks
# prompt = f"You are a smart agent. A question would be asked to you and relevant information would be provided.\
#     Your task is to answer the question and use the information provided. Question - {query}. Relevant Information - {[chunks[index] for index in idx]}"

# # Prompt an LLM with the prompt.
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content(prompt)
# logging.info(f"Response: {response.text}")