## we will make a agentic chatbot with memory

import os
from typing import Dict, TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]
    
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

def process(state:AgentState)->AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    
    state['messages'].append(AIMessage(content = response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state['messages']) 

    return state

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    
    result = agent.invoke({"messages": conversation_history})

    conversation_history = result['messages']
    
    user_input = input('Enter: ')
    
## we will now store all the conversation history of our bot inside a text file so that it does not gets lost when we stop running the code, we can also save it in teh db but it is more simple for quick projects.
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")