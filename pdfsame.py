from typing import TypedDict, List, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY
import pypdf
import os
import glob

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
    temperature=0.1,  # Low temperature for consistent responses
)

# ==================== MULTI-PDF LOADING ====================
def load_all_pdfs_from_folder(folder_name: str = "pdf"):
    """
    Load all PDFs from a specific folder (default: 'pdf' folder)
    The folder should be in the same directory as this Python file
    """
    
    # Get the directory where this Python file is located
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the PDF folder
    pdf_folder = os.path.join(current_folder, folder_name)
    
    # Check if folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ Folder '{folder_name}' not found in: {current_folder}")
        print(f"   Please create a folder named '{folder_name}' and place your PDFs inside.")
        return None, None
    
    # Find all PDF files in the folder
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdf_folder}")
        print(f"   Please place your PDF files in the '{folder_name}' folder.")
        return None, None
    
    print(f"📁 Found {len(pdf_files)} PDF files in '{folder_name}' folder:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")
    
    all_chunks = []
    all_sources = []
    
    for pdf_path in pdf_files:
        print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                page_count = 0
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n{page_text}"
                        page_count += 1
                
                print(f"   ✅ Loaded {page_count} pages")
                
                # Split into chunks
                chunks = chunk_text(text, os.path.basename(pdf_path))
                all_chunks.extend(chunks)
                all_sources.extend([os.path.basename(pdf_path)] * len(chunks))
                
        except Exception as e:
            print(f"   ❌ Error loading {os.path.basename(pdf_path)}: {e}")
    
    print(f"\n✅ Total: {len(all_chunks)} chunks loaded from {len(pdf_files)} PDFs")
    return all_chunks, all_sources

def chunk_text(text: str, source_name: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into chunks with source tracking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            # Add source info at the beginning of each chunk
            chunks.append(f"[Source: {source_name}] {chunk}")
    
    print(f"   ✂️ Split into {len(chunks)} chunks")
    return chunks

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
            # Remove source tag for scoring
            clean_chunk = chunk.replace("[Source: ", "").split("] ", 1)[-1] if "[Source:" in chunk else chunk
            chunk_words = set(clean_chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            
            if score > 0:
                scored_chunks.append((score, idx, chunk))
        
        # Sort by score (highest first)
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        # Get top k chunks
        top_chunks = scored_chunks[:k]
        
        results = []
        sources = []
        for score, idx, chunk in top_chunks:
            results.append(chunk)
            sources.append(self.sources[idx])
        
        return results, sources

# ==================== AGENT CREATION ====================
def create_multi_pdf_agent(folder_name: str = "pdf"):
    """
    Create an agent that can answer questions from multiple PDFs
    in the specified folder (default: 'pdf' folder)
    """
    
    print(f"\n📁 Scanning '{folder_name}' folder for PDF files...")
    
    # Load all PDFs from the folder
    chunks, sources = load_all_pdfs_from_folder(folder_name)
    
    if not chunks:
        print("\n❌ No PDFs found to process.")
        print("   Make sure you have a 'pdf' folder with PDF files in it.")
        print(f"   Folder path: {os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)}")
        return None
    
    # Create retriever
    retriever = MultiPDFRetriever(chunks, sources)
    
    def process(state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        
        # Retrieve relevant chunks from all PDFs
        relevant_chunks, sources = retriever.get_relevant_documents(query, k=5)
        
        # Build context
        context = "\n\n---\n\n".join(relevant_chunks)
        state['context'] = relevant_chunks
        state['sources'] = list(set(sources))  # Unique sources
        
        # Create prompt with source information
        source_info = "\n".join([f"- {src}" for src in set(sources)]) if sources else "None"
        
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
        
        # Clean the response to remove the "Answer:" prefix if present
        answer = response.content
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        state['messages'].append(AIMessage(content=answer))
        
        print(f"\n🤖 AI: {answer}")
        if sources:
            print(f"📚 Sources: {', '.join(set(sources))}")
        
        return state
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node('process', process)
    graph.add_edge(START, 'process')
    graph.add_edge('process', END)
    agent = graph.compile()
    
    return agent

# ==================== BATCH QUESTION TESTING ====================
def test_questions_batch(agent, questions: List[str]):
    """Test multiple questions at once"""
    print("\n" + "=" * 70)
    print("📋 BATCH QUESTION TESTING")
    print("=" * 70)
    
    conversation_history = []
    results = []
    
    for idx, question in enumerate(questions, 1):
        print(f"\n❓ Q{idx}: {question}")
        print("-" * 50)
        
        conversation_history.append(HumanMessage(content=question))
        result = agent.invoke({'messages': conversation_history})
        conversation_history = result['messages']
        
        answer = result['messages'][-1].content
        sources = result.get('sources', [])
        
        results.append({
            'question': question,
            'answer': answer,
            'sources': sources
        })
        
        print(f"🤖 Answer: {answer}")
        if sources:
            print(f"📚 From: {', '.join(sources)}")
    
    return results

# ==================== MAIN PROGRAM ====================
def main():
    print("=" * 70)
    print("📚 MULTI-PDF QUESTION-ANSWERING AGENT")
    print("=" * 70)
    print("\n📁 Looking for PDFs in the 'pdf' folder...")
    
    # Create agent (automatically finds PDFs in 'pdf' folder)
    try:
        agent = create_multi_pdf_agent("pdf")  # Looks for 'pdf' folder
        if not agent:
            # Try alternative folder names if 'pdf' doesn't exist
            print("\n📁 Trying alternative folder names...")
            for folder in ["PDFs", "pdfs", "documents", "docs"]:
                agent = create_multi_pdf_agent(folder)
                if agent:
                    break
        
        if not agent:
            print("\n❌ Could not find any PDFs.")
            print("   Please create a folder named 'pdf' and place your PDF files inside.")
            print(f"   Expected location: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf')}")
            return
        
        print("\n✅ Agent ready! You can ask questions about ALL PDFs in the 'pdf' folder.")
        print("Type 'exit' to quit.")
        print("Type 'batch' to test multiple questions at once.")
        print("Type 'list' to see all loaded PDFs.")
        print("Type 'status' to see folder information.")
        print("-" * 70)
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return
    
    # Pre-defined test questions
    test_questions = [
        "Who is the CEO of Star-Grace IT Consulting?",
        "What services does Star-Grace IT Consulting offer?",
        "Where is the company's head office located?",
        "What awards has the company won?",
        "What is the company's vision for 2030?",
        "What is the company's motto?",
        "How many employees does the company have?",
        "What year was the company founded?",
        "What are the five pillars of the company's philosophy?",
        "What is the company's approach to implementation?"
    ]
    
    conversation_history = []
    
    while True:
        user_input = input("\n❓ You: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'batch':
            print("\n🧪 Running batch test with 10 questions...\n")
            results = test_questions_batch(agent, test_questions)
            print("\n" + "=" * 70)
            print("📊 BATCH TEST SUMMARY")
            print("=" * 70)
            for idx, r in enumerate(results, 1):
                status = "✅ FOUND" if "couldn't find" not in r['answer'].lower() else "❌ NOT FOUND"
                print(f"{idx}. {status} - {r['question'][:50]}...")
                print(f"   Answer: {r['answer'][:100]}...")
                print()
            continue
        
        if user_input.lower() == 'list':
            print("\n📁 PDFs in 'pdf' folder:")
            current_folder = os.path.dirname(os.path.abspath(__file__))
            pdf_folder = os.path.join(current_folder, "pdf")
            if os.path.exists(pdf_folder):
                pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
                if pdf_files:
                    for pdf in pdf_files:
                        size = os.path.getsize(pdf) / 1024  # Size in KB
                        print(f"   - {os.path.basename(pdf)} ({size:.1f} KB)")
                else:
                    print("   No PDF files found.")
            else:
                print("   'pdf' folder not found!")
            continue
        
        if user_input.lower() == 'status':
            print("\n📊 Folder Status:")
            current_folder = os.path.dirname(os.path.abspath(__file__))
            pdf_folder = os.path.join(current_folder, "pdf")
            print(f"   Python file location: {current_folder}")
            print(f"   PDF folder: {pdf_folder}")
            print(f"   Folder exists: {'✅ Yes' if os.path.exists(pdf_folder) else '❌ No'}")
            if os.path.exists(pdf_folder):
                pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
                print(f"   PDF files found: {len(pdf_files)}")
            continue
        
        if not user_input:
            continue
        
        # Regular question
        conversation_history.append(HumanMessage(content=user_input))
        try:
            result = agent.invoke({'messages': conversation_history})
            conversation_history = result['messages']
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()