import os
import csv
import datetime
import logging
import gradio as gr

from dotenv import load_dotenv
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document 
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

load_dotenv(override=True)
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)

MODEL = "gpt-4.1-mini"
DB_NAME = "vector_db"
LLM = ChatOpenAI(temperature=0.5, model_name=MODEL)


COLUMNS_TO_EMBED = ["Ticket ID","Reported By", "Assigned To", "Description", "Project Name", "Status", "timestamp", "Priority"]
COLUMNS_TO_METADATA = ["Ticket ID","Reported By", "Assigned To", "Description", "Project Name", "Status", "timestamp", "Priority"]



def create_documents() -> list[Document]:
    docs = []
    with open("data/bugs/maya_bug_feature_tickets.csv", newline="", encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for i, row in enumerate(csv_reader):
            to_metadata = {col: str(row[col]) for col in COLUMNS_TO_METADATA if col in row}
            values_to_embed = {k: str(row[k]) for k in COLUMNS_TO_EMBED if k in row}
            
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newDoc = Document(page_content=to_embed, metadata=to_metadata)
            docs.append(newDoc)


    splitter = CharacterTextSplitter(separator = "\n",
                                    chunk_size=500, 
                                    chunk_overlap=0,
                                    length_function=len)
    documents = splitter.split_documents(docs)

    return documents    


def create_vector_db() -> Chroma:
    
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_NAME):
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
        logging.info("Vector database already exists")
    else:
        documents = create_documents()
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=DB_NAME)
        logging.info("Vector database created")
    # sample = vectorstore.get(limit=1)
    # print("Current metadata:", sample['metadatas'][0])
    return vectorstore

