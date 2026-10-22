from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY
import pypdf
import os

# ==================== CONFIGURATION ====================
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: List[str]

llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost"},
    temperature=0.7,
)

# ==================== PDF LOADING WITH PYPDF ====================
def load_pdf_with_pypdf(pdf_path: str):
    """Load PDF using pypdf (no langchain-community needed)"""
    print(f"📄 Loading PDF: {pdf_path}")
    
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            print(f"✅ Page {page_num + 1} loaded")
    
    print(f"✅ Total pages: {len(reader.pages)}")
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    print(f"✂️ Split into {len(chunks)} chunks")
    return chunks

class SimpleRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
    
    def get_relevant_documents(self, query: str, k: int = 3):
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:k]]

def create_pdf_agent(pdf_path: str):
    text = load_pdf_with_pypdf(pdf_path)
    chunks = chunk_text(text)
    retriever = SimpleRetriever(chunks)
    
    def process(state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        relevant_chunks = retriever.get_relevant_documents(query, k=3)
        context = "\n\n---\n\n".join(relevant_chunks)
        state['context'] = relevant_chunks
        
        prompt = f"""You are an AI assistant answering questions about a PDF document.

Context from the PDF:
{context}

Question: {query}

Instructions:
1. Answer ONLY with the direct information from the context
2. If the answer is found, give ONLY the answer - no explanations, no "I found this on page X", no extra details
3. If the answer is NOT found, say EXACTLY: "No answer on this information yet, but check later" - nothing else
4. NEVER mention page numbers, chunks, or sections
5. NEVER say "however", "additionally", or provide related information
6. Keep your response to 1-2 sentences maximum
7. Do not repeat the question or provide context about your search

Answer:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state['messages'].append(AIMessage(content=response.content))
        
        print(f"\n🤖 AI: {response.content}")
        # print(f"📚 Used {len(relevant_chunks)} chunks")
        
        return state
    
    graph = StateGraph(AgentState)
    graph.add_node('process', process)
    graph.add_edge(START, 'process')
    graph.add_edge('process', END)
    agent = graph.compile()
    
    return agent

def main():
    print("=" * 60)
    print("📚 PDF Question-Answering Agent")
    print("=" * 60)
    
    pdf_path = input("\n📄 Enter PDF path: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    try:
        agent = create_pdf_agent(pdf_path)
        print("\n✅ Agent ready! Ask questions about the PDF.")
        print("Type 'exit' to quit.\n")
        print("-" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    conversation_history = []
    
    while True:
        user_input = input("\n❓ You: ").strip()
        
        if user_input.lower() == 'exit' or user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({'messages': conversation_history})
        conversation_history = result['messages']

if __name__ == "__main__":
    main()