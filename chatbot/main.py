from langgraph.graph import StateGraph, START, END
from typing import Dict, TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
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
 
def chat_node(state:ChatState):
    
    #take user query from state
    messages = state['messages']
    
    # send to llm
    response = llm.invoke(messages) #we will invoke the llm using the user message and save the response
    
    #response store state
    return {'messages' : [response]}
    
checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)

# initial_state = {
#     'messages': [HumanMessage(content = ' what is the capotal of India?')]
# }
# chatbot.invoke(initial_state)['messages'][-1].content

thread_id = '1'

while True:
    user_message = input('Type here: ')
    
    if user_message .strip().lower() in ['exit', 'quit', 'stop']:
        break
    
    config = {'configurable': {'thread_id': thread_id}}
    
    response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config = config)
    
    print('AI: ', response['messages'][-1].content)
    
    
 