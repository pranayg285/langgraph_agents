import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # Use ChatOpenAI, not AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Setup for Azure Model-as-a-Service (Llama, Mistral, etc.)
llm = ChatOpenAI(
    # Point directly to your custom base URL, dropping /chat/completions
    base_url="https://custom-kliniq-ai-mf-endpoints.services.ai.azure.com/models",
    
    # Pass your Azure API key
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    
    # Put your model name here (e.g., Llama-3-8b-instruct)
    model="Llama-4-Scout-17B-16E-Instruct", 
    
    # Inject the api-version requirement as a URL parameter
    model_kwargs={
        "extra_query": {"api-version": "2024-05-01-preview"}
    }
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return {"messages": state['messages'] + [response]}

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content = user_input)]})
    user_input = input("Enter: ")