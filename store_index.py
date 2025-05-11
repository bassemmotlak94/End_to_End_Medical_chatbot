from src.helper import load_data , text_split ,download_HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

## getting the APIs
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
index_name = "testing"
## loading the data
extracted_data = load_data("Data/")
## splitting Data
text_chunks = text_split(extracted_data)
## preparing the embedding model
embeddings = download_HuggingFaceEmbeddings()
## Storing chunks in vectorDB
vector_store = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)