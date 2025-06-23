import logging
import gradio as gr

from typing import Annotated, TypedDict, Dict, Any
from IPython.display import Image, display
from langchain.docstore.document import Document 
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from csv_checker_tool import create_vector_db
from timestamp_tool import convert_to_timestamp

logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)


VECTORSTORE = create_vector_db()
LLM = ChatOpenAI(model="gpt-4.1-mini")



def call_conversation_chain(text: str) -> list[Document]:
    # return conversation_chain.invoke({"question": text})["answer"]
    return VECTORSTORE.similarity_search(query=text, k=200)

csvtool = Tool(
    name="ticket_board",
    description="A tool for checking the ticket board",
    func=call_conversation_chain,
)

date_to_timestamp_tool = Tool(
    name="date_to_timestamp",
    description="A tool for converting a date to a timestamp",
    func=convert_to_timestamp,
)



tools = [csvtool, date_to_timestamp_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm_with_tools = LLM.bind_tools(tools)


def chatbot(state: State) -> Dict[str, Any]:
    system_message = f"""You are a helpful assistant that can use tools to complete tasks, always use the tools to complete the task, don't make up information.
When you are asked to check the ticket board, you should use the ticket_board tool to check the ticket board, which will return a list of tickets based on the query.

if there is a mention of date or time, you should use the date_to_timestamp tool to get the date and time into a timestamp then use the timestamp value to query the ticket_board againts "Date Created"
timestamp values.

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""
    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True
    
    if not found_system_message:
        messages = [SystemMessage(content=system_message)] + messages
    
    # Invoke the LLM with tools
    response = llm_with_tools.invoke(messages)
    
    # Return updated state
    return {
        "messages": [response],
    }



graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "10"}}

async def chat(user_input: str, history):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content

def main():
    gr.ChatInterface(chat, type="messages").launch()



# config = {"configurable": {"thread_id": "10"}}
# user_input = "Can you get the list of tickets in the tickets board generated after 2024 Christmas , use that timestamp to get the tickets compare against the timestamp value in the tickets board?"
# graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)


if __name__ == "__main__":
    main()
