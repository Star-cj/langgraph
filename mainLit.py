import streamlit as st
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


@st.cache_resource
def build_agent():
    llm = ChatOpenAI(
        model="glm-5.3-flash",  # "mistralai/mistral-7b-instruct:free"
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost"},
        temperature=0.7,
    )

    def process(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        state["messages"].append(AIMessage(content=response.content))
        return state

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()


st.set_page_config(page_title="AI Chat", page_icon="🤖")
st.title("🤖 AI Chat")

agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle new input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": list(st.session_state.messages)})
        reply = result["messages"][-1].content
        st.markdown(reply)

    st.session_state.messages = result["messages"]