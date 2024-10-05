# RAG

## Basic idea:
1. Read the external text file and split it into chunks.
2. Initialize an embedding model.
3. Generate embeddings for each chunk.
4. Generate embedding of the query ( QE).
5. Generate a similarity score betweent QE and each of the Chunk embeddings.
6. Extract top K chunks based on similarity score.
7. Frame a prompt with the query and the top-k chunks.
8. Prompt an LLM with the prompt framed in step-7.

## Hyperparameter tunning
1. What is the ideal chunk size ?
2. Which Embedding models should I use?
3. What should be an ideal value of K?
4. What is an ideal way to store chunk embeddings?
5. Is this LLM good for my use case?
6. Do I need to reframe my prompt?

# Set-up
## Activate virtual environment
### Open terminal and run the following commands
- cd /path/to/your/project
- python -m venv rag_venv  # "rag_venv" is the name of the virtual environment folder
- source rag_venv/bin/activate # "rag_venv" is a folder that will be created when you run this command.

## Install the required dependencies
- pip install --upgrade pip
- pip install -r requirements.txt

## To deactivate the virtual environment
###  run the following command on terminal.
- deactivate