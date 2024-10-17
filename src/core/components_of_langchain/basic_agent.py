# Import relevant functionality
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import sys

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.append(project_root)

# Load environment variables from .env file
load_dotenv()
palm_api_key = os.getenv('PALM_API_KEY', "")
langsmith_api_key = os.getenv('LANGSMITH_API_KEY', "")

# Configure the API key
genai.configure(api_key=palm_api_key)

# loading environment variables for langsmith
LANGCHAIN_TRACING=False # set to true to enable tracing in langsmith application
if LANGCHAIN_TRACING:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=langsmith_api_key
    LANGCHAIN_PROJECT="pr-bumpy-chair-44"

# Create the agent
memory = MemorySaver()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in Seattle")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")