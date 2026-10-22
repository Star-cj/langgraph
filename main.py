from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY

class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

llm = ChatOpenAI(
    model="glm-5.3-flash",     # "mistralai/mistral-7b-instruct:free", 
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",  
    },
    temperature=0.7,
)

def process(state:AgentState)->AgentState:
    response=llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))
    print(f'\nAI:{response.content}')
    return state

graph=StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
agent=graph.compile()

conversation_history=[]

user_input=input('Enter: ')

while user_input !='exit':
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({'messages':conversation_history})

    print(result['messages'])
    conversation_history=result['messages']

    user_input=input('Enter: ')
    
