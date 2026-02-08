import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
import time

from google import genai
from google.genai import types

# ---------------------------------------------------------------------
# 1. Setup Environment
load_dotenv()


''' ****************************************************************
DO NOT RUN THIS CODE BLOCK AGAIN, AS DURING 1st RUN IT HAS loaded pdf, done chunking, STORED THE DOCUMENTS IN PINECONE VECTOR DATABASE, IF YOU RUN THIS CODE BLOCK AGAIN, IT WILL UPLOAD THE SAME DOCUMENTS AGAIN TO PINECONE AND YOU WILL GET DUPLICATE DOCUMENTS IN PINECONE VECTOR DATABASE, SO PLEASE DO NOT RUN THIS CODE BLOCK AGAIN, AS IT HAS STORED THE DOCUMENTS IN PINECONE VECTOR DATABASE.
RUN THIS BLOCK AGAIN IF AND ONLY IF YOU WANT TO UPLOAD SOME NEW DOCUMENTS TO PINECONE VECTOR DATABASE, OTHERWISE PLEASE DO NOT RUN THIS BLOCK AGAIN.

# ---------------------------------------------------------------------
# 2. Load PDF
loader = PyPDFLoader("./Dsa50pages.pdf")
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} pages.")

# ---------------------------------------------------------------------
# # 3. Split Text , creating chunks of file of 1000 characters with an overlap of 200 characters to maintain context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunked_docs = text_splitter.split_documents(raw_docs)
print(f"Created {len(chunked_docs)} chunks.")
'''
# ---------------------------------------------------------------------
# 4. Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1024  #because on pinecone vectordb we have choosen 1024 dimensionality for our index, but this google gemini embedding model returns 3072 dimensionality by default, so we need to set output_dimensionality to 1024 to match our pinecone index dimensionality. If you don't set this, you will get an error when trying to upload to pinecone because of dimensionality mismatch.
)

# ---------------------------------------------------------------------
# 5. Initialize Pinecone and vector store created on pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
# Initialize an empty vector store first which we have created on pinecone or it could be already existing one for vector search purpose if its already created
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
'''
# ---------------------------------------------------------------------
# 6. Embedding the chunks using google gemini-embedding-001 model and uploading to Pinecone in batches to avoid hitting the rate limit of 100 requests per minute for the free tier embedding model. Each batch will contain 5 chunks, and we will wait for 10 seconds between batches to reset the "per minute" quota.
# This below line can upload your documents to Pinecone in one go, but embedding model is free tier and has a rate limit of 100 requests per minute, so we will upload documents in batches to avoid hitting the rate limit.
# vector_store = PineconeVectorStore.from_documents(chunked_docs, embeddings, index_name=index_name) - it hits gemini-embedding-model rate limit of 100/per minute, so we will upload documents in batches to avoid hitting the rate limit.


# Upload in small batches with a delay
batch_size = 5  # Small batches to stay under free tier limits
print(f"Starting batch upload... Total chunks: {len(chunked_docs)}")

for i in range(0, len(chunked_docs), batch_size):
    batch = chunked_docs[i : i + batch_size]
    vector_store.add_documents(batch)
    print(f"Uploaded chunks {i} to {i + len(batch)}")
    # Wait 5-10 seconds between batches to reset your "per minute" quota
    time.sleep(15) 

print("Upload complete!")
print("Documents uploaded to Pinecone successfully.")

**************************************************************************
'''
# ----------------------------------------------------------------------
# creating LLM connection for Gemini LLM 
client = genai.Client(api_key="Your_GEMINI_API_KEY_Here_fromGoogleStudio")
chat = client.chats.create(
    model="gemini-3-flash-preview"
)

instruction_template = """
You are a Data Structure and Algorithm Expert. 
Use the following context to answer the user's question. 
If the answer isn't in the context, say you don't know.

Context:{context_for_LLM}

Question: {prompt_by_user}
"""
context_template = ChatPromptTemplate.from_template(instruction_template)

# RAG Process - Retrieval Augmentation Generation Process
def RAG_Process(prompt: str) -> str:
    # RAG - Retrieval: this prompt will be converted into vector using same embedding model the vector_store is using and then it will run similarity search on this vectorDB to find k most similar vectors/data already stored there
    search_results = vector_store.similarity_search(prompt, k=4)

    # RAG - Augmentaion of LLM Model with "actual prompt + retrieved context from vectorDB", this is the step where we are giving the retrieved context from vectorDB to LLM along with actual user prompt, so that LLM can use this retrieved context to generate better answer for user prompt
    retrived_similar_results = "\n\n---\n\n".join([doc.page_content for doc in search_results])
    new_context_for_LLM = context_template.format(context_for_LLM=retrived_similar_results, prompt_by_user=prompt)

    # RAG - Generation : This is the step where LLM will generate answer for user prompt by using the retrieved context from vectorDB, here we are giving instruction to LLM to answer only based on the retrieved context from vectorDB, so that it will not hallucinate and will give answer only based on the retrieved context from vectorDB
    response = chat.send_message(new_context_for_LLM)
    return response.text


# # 7. Main Loop for user interaction
def run():
    while True:
        prompt = input("\nEnter your prompt: ")
        reply = RAG_Process(prompt)
        print("=> ", reply)
        print("\n-------------------------------------------------------------------------------\n\n")


run()
