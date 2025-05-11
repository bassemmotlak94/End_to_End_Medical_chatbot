from flask import Flask,render_template,request
from src.helper import download_HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
embeddings = download_HuggingFaceEmbeddings()
index_name ="testing"
docsearch = PineconeVectorStore.from_existing_index(index_name,embeddings)
llm = ChatGroq(model = "deepseek-r1-distill-llama-70b"
               ,api_key= groq_key)
prompt_template=PromptTemplate(template=template,
                               input_variables=["context","question"],
)
chain_type_kwargs={"prompt":prompt_template}
qa = RetrievalQA.from_chain_type(llm=llm,chain_type_kwargs=chain_type_kwargs,
           retriever = docsearch.as_retriever(),
            chain_type="stuff" )

@app.route("/")
def index():
 return render_template("chat.html")
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])


if __name__=='__main__' :
  app.run(debug=True)
