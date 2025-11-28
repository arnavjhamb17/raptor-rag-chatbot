import os
import dotenv
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI environment variables
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_version = os.environ.get("AZURE_OPENAI_VERSION")

if not (azure_api_key and azure_endpoint and azure_deployment):
    raise ValueError("Azure OpenAI environment variables are not set. Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME in your .env file.")

llm = AzureChatOpenAI(
    api_version=azure_version,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    """
    Generate embeddings for a list of texts using HuggingFaceEmbeddings.
    Returns a numpy array of embeddings.
    """
    return embeddings.embed_documents(texts)

api_call_count = 0

def summarize_cluster(text: str) -> str:
    """
    Summarizes the provided cluster text using a language model.

    Parameters:
    - text: The text content of a cluster (joined from all its documents).

    Returns:
    - A string containing the summary.
    """
    global api_call_count
    api_call_count += 1
    print(f"[DEBUG] summarize_cluster call #{api_call_count}, input length (chars): {len(text)}")
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that summarizes clusters of related technical documentation.

Cluster Content:
{text}

Please write a clear, informative, and concise summary of the above cluster.
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})
