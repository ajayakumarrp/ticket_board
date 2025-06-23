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


COLUMNS_TO_EMBED = ["Ticket ID","Reported By", "Assigned To", "Description", "Project Name", "Status", "timestamp"]
COLUMNS_TO_METADATA = ["Ticket ID","Reported By", "Assigned To", "Description", "Project Name", "Status", "timestamp"]



def create_documents() -> list[Document]:
    docs = []
    with open("data/bugs/maya_bug_feature_tickets.csv", newline="", encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            # to_metadata = {col: str(row[col]) for col in COLUMNS_TO_METADATA if col in row}
            
            to_metadata = {}
            for col in COLUMNS_TO_METADATA:
                if col in row:
                    # if col == "Date Created":
                    #     date = float(datetime.datetime.strptime(row[col], "%Y-%m-%d").timestamp())
                    #     to_metadata.update({col: date})
                    # else:
                    to_metadata.update({col: row[col]})
                    
            # values_to_embed = {k: str(row[k]) for k in COLUMNS_TO_EMBED if k in row}
            
            values_to_embed = {}
            for k in COLUMNS_TO_EMBED:
                if k in row:
                    # if k == "Date Created":
                    #     date = str(datetime.datetime.strptime(row[k], "%Y-%m-%d").timestamp())
                    #     values_to_embed.update({k: date})
                    # else:
                    values_to_embed.update({k: row[k]})


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
        # Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        documents = create_documents()
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=DB_NAME)

    # sample = vectorstore.get(limit=1)
    # print("Current metadata:", sample['metadatas'][0])
    return vectorstore



# def create_conversation_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:

#     # Metadata schema based on the values on the CSV
#     metadata_field_info = [
#         AttributeInfo(
#             name="Ticket ID",
#             description="ID for the ticket, it is a unique identifier for the ticket",
#             type="string",
#         ),
#         AttributeInfo(
#             name="Reported By",
#             description="Who reported the ticket or created the ticket.",
#             type="string",
#         ),
#         AttributeInfo(
#             name="Assigned To",
#             description="To whom the ticket is assigned to, it can be a person or a team",
#             type="string",
#         ),
#         AttributeInfo(
#             name="Description", 
#             description="Description of the ticket explaining what the issue is or what feature is wanted, it is a detailed description of the ticket", type="string"
#         ),
#         AttributeInfo(
#             name="Project Name", 
#             description="Name of the project to which the ticket belong to, it is the name of the project to which the ticket belong to", 
#             type="string"
#         ),
#         AttributeInfo(
#             name="Status", 
#             description="Current status of the ticket, can be 'On Progress', 'Yet toStart', 'On Hold', 'Done', it is the current status of the ticket", 
#             type="string"
#         ),
#         AttributeInfo(
#             name="Date Created", 
#             description="Timestamp of Date on which the ticket was created, in Timestamp format not date format, it is called timestamp in the data", 
#             type="string"
#         ),
#     ]


#     document_content_description = "Ticket in an animation studio's bug/feature tracker"

#     retriever = SelfQueryRetriever.from_llm(
#         llm=LLM,
#         vectorstore=vectorstore,
#         document_contents=document_content_description,
#         metadata_field_info=metadata_field_info,
#         verbose=True,
#         search_kwargs={"k": 2000},)

#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(llm=LLM, retriever=retriever, memory=memory)

#     return conversation_chain


