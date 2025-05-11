from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
def load_data(data):
 loader = DirectoryLoader(data,
                          glob="*.pdf",
                          loader_cls=PyPDFLoader

 )
 documents = loader.load()
 return documents
def text_split(extracted_data):
 text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=20)
 text_chunks= text_splitter.split_documents(extracted_data)
 return text_chunks
def download_HuggingFaceEmbeddings():
  embeddings =HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embeddings