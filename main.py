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
from tools.csv_checker_tool import create_vector_db
from tools.timestamp_tool import convert_to_timestamp
from tools.user_check_tool import get_current_user

logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)


VECTORSTORE = create_vector_db()
LLM = ChatOpenAI(model="gpt-4.1-mini",seed=42)
CONFIG = {"configurable": {"thread_id": "10"}}

SYSTEM_MESSAGE = f"""You are a helpful assistant that can use tools to complete tasks, always use the tools to complete the task,
    don't make up information which is not present in the ticket board. When you are asked to check the ticket board, you should use the ticket_board tool to check the ticket board,
    which will return a list of tickets based on the query.

    if there is a mention of date or time, you should use the date_to_timestamp tool to get the date and time into a timestamp then use the timestamp value to query the ticket_board
    againts "Date Created" timestamp values. don't return ticket which are not present in the ticket board. don't return timestamp values, return the date in the format of YYYY-MM-DD.

    if the user mentions the current user or as me, you should always use the get_current_user tool to get the current user ID and query it against the "Reported By"
    column in the ticket board or "Assigned To" column in the ticket board, based on what the user is asking, don't make up the user ID or return someone else ticket as this user.

    If you are asked to get tickets based on priority check the priority column in the ticket board. if ticket has priority 1 it is high priority,
    if ticket has priority 2 it is not high priority but above medium priority, if ticket has priority 3 it is medium priority,
    if the ticket has priority 4 it is not medium priority but above low priority, if ticket has priority 5 it is low priority.

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer. always return the answer when you are sure about the answer.
    """



def call_conversation_chain(text: str) -> list[Document]:
    return VECTORSTORE.similarity_search(query=text, k=100)


def create_tools() -> list[Tool]:
    tools = []
    csvtool = Tool(
        name="ticket_board",
        description="A tool for checking the ticket board",
        func=call_conversation_chain,
    )
    tools.append(csvtool)

    date_to_timestamp_tool = Tool(
        name="date_to_timestamp",
        description="A tool for converting a date to a timestamp",
        func=convert_to_timestamp,
    )
    tools.append(date_to_timestamp_tool)

    user_tool = Tool(
        name="get_current_user",
        description="A tool for getting the current user",
        func=get_current_user,
    )
    tools.append(user_tool)

    return tools


LLM_WITH_TOOLS = LLM.bind_tools(create_tools())


class State(TypedDict):
    messages: Annotated[list, add_messages]


def ticket_board_agent(state: State) -> Dict[str, Any]:

    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = SYSTEM_MESSAGE
            found_system_message = True
    
    if not found_system_message:
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
    
    # Invoke the LLM with tools
    response = LLM_WITH_TOOLS.invoke(messages)
    
    # Return updated state
    return {
        "messages": [response],
    }


def create_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("ticket_board_agent", ticket_board_agent)
    graph_builder.add_node("tools", ToolNode(tools=create_tools()))

    graph_builder.add_conditional_edges("ticket_board_agent", tools_condition, "tools")

    graph_builder.add_edge("tools", "ticket_board_agent")
    graph_builder.add_edge(START, "ticket_board_agent")

    graph = graph_builder.compile()
    # display(Image(graph.get_graph().draw_mermaid_png()))
    return graph

async def chat(user_input: str, history: list[tuple[str, str]]) -> str:
    graph = create_graph()
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=CONFIG)
    return result["messages"][-1].content

def main() -> None:
    gr.ChatInterface(chat, title="Ticket Board", theme=gr.themes.Soft(), type="messages").launch(inbrowser=True)

    ## To test the graph
    # graph = create_graph()
    # user_input = "Can you please list of 10 recent priority 2 tickets assigned to me? along with dates it got created"
    # result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=CONFIG)
    # print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
