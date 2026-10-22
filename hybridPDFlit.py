import streamlit as st
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY
import pypdf
import os
import glob

st.set_page_config(page_title="Multi-PDF Q&A", page_icon="📚", layout="centered")

TEST_QUESTIONS = [
    "Who is the CEO of Star-Grace IT Consulting?",
    "What services does Star-Grace IT Consulting offer?",
    "Where is the company's head office located?"
]

# ==================== CONFIGURATION ====================
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: List[str]
    sources: List[str]  # Track which PDF each chunk came from

llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost"},
    temperature=0.1,
    max_tokens=500
)

# ==================== MULTI-PDF LOADING ====================
def chunk_text(text: str, source_name: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into chunks with source tracking"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(f"[Source: {source_name}] {chunk}")
    return chunks

def load_pdfs(folder_name: str = "pdf"):
    """Load all PDFs from the folder next to this file.
    Returns chunks, sources, loaded file names, and errors."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(base_dir, folder_name)

    all_chunks, all_sources, loaded, errors = [], [], [], []

    if not os.path.isdir(pdf_folder):
        errors.append(f"Folder '{folder_name}' not found")
        return all_chunks, all_sources, loaded, errors

    for pdf_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        name = os.path.basename(pdf_path)
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(text, name)
            all_chunks.extend(chunks)
            all_sources.extend([name] * len(chunks))
            loaded.append(name)
        except Exception as e:
            errors.append(f"{name}: {e}")

    return all_chunks, all_sources, loaded, errors

# ==================== RETRIEVER ====================
class MultiPDFRetriever:
    def __init__(self, chunks: List[str], sources: List[str]):
        self.chunks = chunks
        self.sources = sources

    def get_relevant_documents(self, query: str, k: int = 5):
        """Retrieve relevant chunks from all PDFs"""
        query_words = set(query.lower().split())
        scored_chunks = []

        for idx, chunk in enumerate(self.chunks):
            clean_chunk = chunk.replace("[Source: ", "").split("] ", 1)[-1] if "[Source:" in chunk else chunk
            chunk_words = set(clean_chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, idx, chunk))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])

        results, sources = [], []
        for score, idx, chunk in scored_chunks[:k]:
            results.append(chunk)
            sources.append(self.sources[idx])

        return results, sources

# ==================== AGENT (BUILT ONCE AND CACHED) ====================
@st.cache_resource(show_spinner="Loading PDFs and building agent...")
def build_agent(folder_name: str = "pdf"):
    chunks, sources, loaded, errors = load_pdfs(folder_name)
    info = {"files": loaded, "chunks": len(chunks), "errors": errors}

    if not chunks:
        return None, info

    retriever = MultiPDFRetriever(chunks, sources)

    def process(state: AgentState) -> AgentState:
        query = state["messages"][-1].content

        relevant_chunks, pdf_sources = retriever.get_relevant_documents(query, k=5)
        context = "\n\n---\n\n".join(relevant_chunks)
        state["context"] = relevant_chunks
        state["sources"] = list(set(pdf_sources))

        source_info = "\n".join(f"- {src}" for src in set(pdf_sources)) if pdf_sources else "None"

        prompt = f"""You are a strict AI assistant that answers questions based ONLY on the provided PDF content from multiple documents.

Context from PDFs:
{context}

Sources used:
{source_info}

Question: {query}

STRICT RULES:
1. If the answer is found in ANY of the documents: Return ONLY the direct answer with the source.
   Format: "Answer: [your answer] (Source: [filename])"
2. If the answer is NOT found: Return EXACTLY "I couldn't find that information in any of the PDFs."
3. NEVER provide additional context, explanations, or clarifications.
4. If multiple sources have the answer, mention the most relevant one.
5. Keep response to 1-2 sentences maximum.

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])

        answer = response.content
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()

        state["messages"].append(AIMessage(content=answer))
        return state

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile(), info

# ==================== STREAMLIT UI ====================
st.title("📚 Multi-PDF Q&A")
st.caption("Ask a question and get answers from all the PDFs in the 'pdf' folder.")

agent, pdf_info = build_agent("pdf")

if "history" not in st.session_state:
    st.session_state.history = []   # LangChain messages sent to the agent
if "chat" not in st.session_state:
    st.session_state.chat = []      # Display data for the UI
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

def ask(question: str):
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.history.append(HumanMessage(content=question))

    with st.spinner("Searching PDFs..."):
        result = agent.invoke({"messages": list(st.session_state.history)})

    st.session_state.history = result["messages"]
    st.session_state.chat.append({
        "role": "assistant",
        "content": result["messages"][-1].content,
        "sources": result.get("sources", []),
    })

def run_batch():
    """Run the test questions with their own history, like the original batch command."""
    results = []
    history = []
    progress = st.progress(0.0, text="Running batch test...")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        history.append(HumanMessage(content=q))
        result = agent.invoke({"messages": list(history)})
        history = result["messages"]
        answer = result["messages"][-1].content
        results.append({
            "question": q,
            "answer": answer,
            "sources": result.get("sources", []),
            "found": "couldn't find" not in answer.lower(),
        })
        progress.progress(i / len(TEST_QUESTIONS), text=f"Question {i} of {len(TEST_QUESTIONS)}")
    progress.empty()
    st.session_state.batch_results = results

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("📚 PDF Knowledge Base")
    if pdf_info["files"]:
        st.write(f"{len(pdf_info['files'])} PDF(s), {pdf_info['chunks']} chunks")
        for f in pdf_info["files"]:
            st.write(f"- {f}")
    else:
        st.error("No PDFs loaded.")
    for err in pdf_info["errors"]:
        st.warning(err)

    if st.button("🔄 Reload PDFs"):
        build_agent.clear()
        st.rerun()

    st.divider()
    run_batch_clicked = st.button("🧪 Run batch test", disabled=agent is None)

    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.session_state.chat = []
        st.session_state.batch_results = None
        st.rerun()

    if st.button("🛑 End session"):
        st.session_state.clear()
        st.warning("Session ended. Close this tab to finish.")
        st.stop()

# ---------- Main area ----------
if agent is None:
    st.error("Could not find any PDFs. Create a 'pdf' folder next to app.py and add your PDF files.")
    st.stop()

# Handle actions first so the new messages show in this same run
if run_batch_clicked:
    run_batch()

user_input = st.chat_input("Ask a question about your PDFs...")
if user_input:
    ask(user_input.strip())

# Batch results
if st.session_state.batch_results:
    with st.expander("📊 Batch test summary", expanded=True):
        found = sum(r["found"] for r in st.session_state.batch_results)
        st.write(f"Found {found} of {len(st.session_state.batch_results)} answers in the PDFs.")
        for i, r in enumerate(st.session_state.batch_results, 1):
            icon = "✅" if r["found"] else "❌"
            st.markdown(f"**{i}. {icon} {r['question']}**")
            st.write(r["answer"])
            if r["sources"]:
                st.caption("From: " + ", ".join(sorted(set(r["sources"]))))

# Conversation
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("📚 Sources: " + ", ".join(sorted(set(msg["sources"]))))