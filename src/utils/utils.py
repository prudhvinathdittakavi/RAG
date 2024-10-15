import numpy as np
import llama_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import math
import os
from dotenv import load_dotenv
import PyPDF2
import logging

# Configuring the logger
# different logging levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Creating a custom logger
logger = logging.getLogger(__name__)

def read_file_based_on_type(file_path):
    """
    Function to identify the file type from the file path and read the file accordingly.
    
    Parameters:
        file_path (str): The path of the file to read.
        
    Returns:
        str: The content of the file as a string.
    
    Raises:
        ValueError: If the file type is unsupported.
    """
    
    # Identify the file type by splitting the file path and checking the extension
    file_type = file_path.split(".")[-1].lower()
    
    # Read the text file based on the file type
    if file_type == "pdf":
        # If the file is a PDF, call a function to extract text from the PDF
        text_file = extract_text_from_pdf(file_path)
    elif file_type == "txt":
        # If the file is a text file, call a function to read the text file
        text_file = read_text_file(file_path)
    else:
        # Raise an error if the file type is unsupported
        raise ValueError("Unsupported file type.")
    
    # Return the extracted text
    return text_file

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract the text from a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The text extracted from the PDF file.
    """

    # Initialize an empty string to store the extracted text
    text = ""

    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, "rb") as pdf_file:
            # Create a PyPDF2.PdfReader object from the PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Loop over all the pages in the PDF file
            for page_num in range(len(pdf_reader.pages)):
                # Get the current page object
                page = pdf_reader.pages[page_num]

                # Extract the text from the current page and add it to the total text
                text += page.extract_text()

    except Exception as e:
        # Log the error and return an empty string
        logger.error(f"Error extracting text from PDF: {e}")
        raise e

    # Return the total text extracted from the PDF file
    return text

def read_text_file(file_path: str):
    """
    Function to read the contents of a text file

    Args:
        file_path (str): The path to the file as a string

    Returns:
        str: The content of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise e

def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Break a large text into smaller chunks of a specified size.

    Args:
        text (str): The text string to be chunked.
        chunk_size (int): The size of each chunk (number of characters).

    Returns:
        list[str]: A list of text chunks (each chunk is a string).
    """
    try:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        logger.error(f"Error breaking text into chunks: {e}")
        raise e
    return chunks

def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Function to initialize and return a pre-trained embedding model.

    This function uses the HuggingFaceEmbedding class to load a specific model.
    The model is BAAI/bge-small-en-v1.5, which is a small English language model
    that is suitable for generating embeddings for text.

    Returns:
        An embedding model object that can generate embeddings for text
    """
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise e
    return embed_model

def get_embeddings(embed_model, text: str):
    """
    Function to get the text embeddings for a given text using a provided embedding model.

    Args:
        embed_model (HuggingFaceEmbedding): The pre-trained embedding model.
        text (str): The input text string for which embeddings are required.

    Returns:
        list[float]: The embedding vector as a list of floats.
    """
    try:
        # Get the text embeddings for the given text using the provided embedding model
        embeddings = embed_model.get_text_embedding(text)
    except Exception as e:
        # Log the error and return an empty list
        logger.error(f"Error getting embeddings: {e}")
        raise e
    return embeddings

def dot_product(vec1, vec2):
    """
    Function to compute the dot product of two vectors.

    The dot product of two vectors is the sum of the products of the corresponding
    entries of the two sequences of numbers. It is a measure of how similar two
    vectors are.

    Args:
        vec1 (list[float]): The first vector.
        vec2 (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    return sum(a * b for a, b in zip(vec1, vec2))

def magnitude(vec):
    """
    Function to calculate the magnitude (Euclidean norm) of a vector.
    
    The magnitude of a vector is a measure of its size. It is calculated as
    the square root of the sum of the squares of the elements of the vector.
    
    Args:
        vec (list[float]): The input vector (list of floats)
    
    Returns:
        float: The magnitude of the vector (a float)
    """
    return math.sqrt(sum(v**2 for v in vec))

def cosine_similarity(vec1, vec2):
    """
    Function to compute the cosine similarity between two vectors.

    The cosine similarity is a measure of how similar two vectors are.
    It is defined as the dot product of the two vectors divided by the product
    of their magnitudes. The cosine similarity is a value between -1 and 1,
    where 1 means the vectors are identical and -1 means they are opposite.

    Args:
        vec1 (list[float]): The first vector.
        vec2 (list[float]): The second vector.

    Returns:
        float: The cosine similarity score (a float between -1 and 1).
    """
    try:
        # Calculate the dot product of the two vectors
        dot_prod = dot_product(vec1, vec2)

        # Calculate the magnitudes of the two vectors
        mag_vec1 = magnitude(vec1)
        mag_vec2 = magnitude(vec2)

        # Handle division by zero
        if mag_vec1 == 0 or mag_vec2 == 0:
            return 0

        # Calculate the cosine similarity
        return dot_prod / (mag_vec1 * mag_vec2)
    except ZeroDivisionError as e:
        # Log the error
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0
