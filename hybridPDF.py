from typing import TypedDict, List, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from key import OPENROUTER_API_KEY
import pypdf
import os
import glob
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import quote_plus

# ==================== CONFIGURATION ====================
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: List[str]
    sources: List[str]
    search_type: str  
    confidence: float  

llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost"},
    temperature=0.1,
)

# ==================== PDF LOADING ====================
def load_all_pdfs_from_folder(folder_name: str = "pdf"):
    """Load all PDFs from the specified folder"""
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(current_folder, folder_name)
    
    if not os.path.exists(pdf_folder):
        print(f"❌ Folder '{folder_name}' not found")
        return None, None
    
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{folder_name}'")
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
            chunks.append(f"[Source: {source_name}] {chunk}")
    
    print(f"   ✂️ Split into {len(chunks)} chunks")
    return chunks

# ==================== INTERNET SEARCH ====================
class InternetSearch:
    """Handle internet search with confidence scoring"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_web(self, query: str, max_results: int = 3):
        """Search the web for information"""
        print(f"🌐 Searching internet for: {query}")
        
        try:
            # Use DuckDuckGo HTML search (free, no API key needed)
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                print("   ⚠️ Search failed, trying alternative...")
                return self._search_fallback(query)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            result_elements = soup.find_all('a', class_='result__a')
            snippets = soup.find_all('a', class_='result__snippet')
            
            for i, (result, snippet) in enumerate(zip(result_elements[:max_results], snippets[:max_results])):
                title = result.get_text(strip=True) if result else ""
                link = result.get('href', '') if result else ""
                description = snippet.get_text(strip=True) if snippet else ""
                
                if title and description:
                    results.append({
                        'title': title,
                        'link': link,
                        'description': description,
                        'full_text': f"{title}. {description}"
                    })
            
            print(f"   ✅ Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"   ⚠️ Search error: {e}")
            return self._search_fallback(query)
    
    def _search_fallback(self, query: str):
        """Fallback search using different method"""
        try:
            # Try Google search fallback
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('div', class_='g')[:3]:
                    title_elem = result.find('h3')
                    desc_elem = result.find('div', class_='IsZvec')
                    
                    if title_elem and desc_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'link': '',
                            'description': desc_elem.get_text(strip=True),
                            'full_text': f"{title_elem.get_text(strip=True)}. {desc_elem.get_text(strip=True)}"
                        })
                
                print(f"   ✅ Found {len(results)} results (fallback)")
                return results
            
            return []
            
        except Exception as e:
            print(f"   ❌ Fallback search failed: {e}")
            return []
    
    def get_answer_with_confidence(self, query: str, search_results: List[Dict]) -> Dict:
        """Evaluate search results and return answer with confidence score"""
        
        if not search_results:
            return {
                'answer': None,
                'confidence': 0.0,
                'sources': []
            }
        
        # Combine all search result text
        combined_text = "\n\n".join([r['full_text'] for r in search_results])
        
        # Use LLM to evaluate confidence
        prompt = f"""You are a strict information evaluator. Given the search results and question, determine:
1. If the answer can be found in the search results
2. Your confidence level (0-100%) in the answer

Search Results:
{combined_text}

Question: {query}

Respond in this exact format:
CONFIDENCE: [0-100]
ANSWER: [your answer if found, or "NOT FOUND" if not]

Be strict - only answer if the information is clearly present."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Parse confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', content)
            confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0
            
            # Parse answer
            answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', content, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else None
            
            if answer and answer.upper() != "NOT FOUND" and confidence >= 0.75:
                sources = [{'title': r['title'], 'link': r['link']} for r in search_results if r['title']]
                return {
                    'answer': answer,
                    'confidence': confidence,
                    'sources': sources[:2]  
                }
            
            return {
                'answer': None,
                'confidence': confidence,
                'sources': []
            }
            
        except Exception as e:
            print(f"   ❌ Error evaluating answer: {e}")
            return {
                'answer': None,
                'confidence': 0.0,
                'sources': []
            }

# ==================== RETRIEVER ====================
class MultiPDFRetriever:
    def __init__(self, chunks: List[str], sources: List[str]):
        self.chunks = chunks
        self.sources = sources
    
    def get_relevant_documents(self, query: str, k: int = 5):
        """Retrieve relevant chunks from PDFs"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for idx, chunk in enumerate(self.chunks):
            clean_chunk = chunk.replace("[Source: ", "").split("] ", 1)[-1] if "[Source:" in chunk else chunk
            chunk_words = set(clean_chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            
            if score > 0:
                scored_chunks.append((score, idx, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        top_chunks = scored_chunks[:k]
        
        results = []
        sources = []
        for score, idx, chunk in top_chunks:
            results.append(chunk)
            sources.append(self.sources[idx])
        
        return results, sources

# ==================== AGENT CREATION ====================
def create_hybrid_agent(folder_name: str = "pdf"):
    """Create a hybrid agent that first searches PDFs, then internet"""
    
    print(f"\n📁 Scanning '{folder_name}' folder for PDF files...")
    
    # Load PDFs
    chunks, sources = load_all_pdfs_from_folder(folder_name)
    
    if not chunks:
        print("\n⚠️ No PDFs found. Agent will only use internet search.")
        chunks = []
        sources = []
    
    retriever = MultiPDFRetriever(chunks, sources) if chunks else None
    internet_search = InternetSearch()
    
    def process(state: AgentState) -> AgentState:
        query = state['messages'][-1].content
        confidence = 0.0
        search_type = "none"
        final_answer = None
        
        print(f"\n🔍 Searching for: {query}")
        print("-" * 50)
        
        # STEP 1: Search PDFs first
        print("📚 Step 1: Checking PDFs...")
        
        if retriever and chunks:
            relevant_chunks, sources = retriever.get_relevant_documents(query, k=5)
            
            if relevant_chunks:
                context = "\n\n---\n\n".join(relevant_chunks)
                state['context'] = relevant_chunks
                state['sources'] = list(set(sources))
                
                # Evaluate PDF answer confidence
                prompt = f"""You are a strict evaluator. Based on the context, answer the question.
                
Context:
{context}

Question: {query}

Rules:
1. If the answer is clearly in the context, provide it
2. If not, say "NOT FOUND"
3. Be precise and concise

Answer:"""
                
                try:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    pdf_answer = response.content.strip()
                    
                    if pdf_answer and pdf_answer.upper() != "NOT FOUND":
                        # Check if it's a real answer (not just saying not found)
                        not_found_phrases = ["not found", "couldn't find", "don't have", "no information"]
                        is_not_found = any(phrase in pdf_answer.lower() for phrase in not_found_phrases)
                        
                        if not is_not_found:
                            # Evaluate confidence
                            confidence_prompt = f"""Rate your confidence (0-100) that this answer is correct based on the context.

Question: {query}
Answer: {pdf_answer}

Respond with only a number between 0-100."""
                            
                            conf_response = llm.invoke([HumanMessage(content=confidence_prompt)])
                            try:
                                confidence = int(re.search(r'\d+', conf_response.content).group()) / 100
                            except:
                                confidence = 0.85  # Default high confidence
                            
                            if confidence >= 0.75:
                                final_answer = pdf_answer
                                search_type = "pdf"
                                print(f"   ✅ Found in PDFs (Confidence: {confidence:.0%})")
                                state['confidence'] = confidence
                                state['search_type'] = search_type
                                state['messages'].append(AIMessage(content=final_answer))
                                
                                print(f"\n🤖 AI: {final_answer}")
                                print(f"📚 Source: PDF (Confidence: {confidence:.0%})")
                                return state
                except Exception as e:
                    print(f"   ⚠️ PDF evaluation error: {e}")
        
        # STEP 2: If not found in PDFs or low confidence, search internet
        if not final_answer or confidence < 0.75:
            print("📚 Step 2: Not found in PDFs, searching internet...")
            
            # Search the internet
            search_results = internet_search.search_web(query)
            
            if search_results:
                # Evaluate internet answer with confidence
                result = internet_search.get_answer_with_confidence(query, search_results)
                
                if result['answer'] and result['confidence'] >= 0.75:
                    final_answer = result['answer']
                    search_type = "internet"
                    confidence = result['confidence']
                    
                    state['sources'] = [s['title'] for s in result['sources'] if s['title']]
                    state['context'] = [s['description'] for s in result['sources']]
                    
                    print(f"   ✅ Found on internet (Confidence: {confidence:.0%})")
                    if state['sources']:
                        print(f"   📎 Sources: {', '.join(state['sources'][:2])}")
                    
                    state['confidence'] = confidence
                    state['search_type'] = search_type
                    state['messages'].append(AIMessage(content=final_answer))
                    
                    print(f"\n🤖 AI: {final_answer}")
                    print(f"📚 Source: Internet (Confidence: {confidence:.0%})")
                    return state
                else:
                    print(f"   ⚠️ Insufficient confidence: {result['confidence']:.0%}")
                    confidence = result['confidence']
            else:
                print("   ❌ No internet results found")
        
        # STEP 3: If not found with sufficient confidence
        if not final_answer or confidence < 0.75:
            final_answer = "I couldn't find reliable information about this query. Please check back later or try rephrasing your question."
            search_type = "none"
            confidence = 0.0
            
            print("   ❌ Answer not found with sufficient confidence")
            print(f"\n🤖 AI: {final_answer}")
        
        state['messages'].append(AIMessage(content=final_answer))
        state['confidence'] = confidence
        state['search_type'] = search_type
        return state
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node('process', process)
    graph.add_edge(START, 'process')
    graph.add_edge('process', END)
    agent = graph.compile()
    
    return agent

# ==================== MAIN PROGRAM ====================
def main():
    print("=" * 70)
    print("🌐 HYBRID PDF + INTERNET QUESTION-ANSWERING AGENT")
    print("=" * 70)
    print("\n📌 How it works:")
    print("   1️⃣ First searches PDFs for the answer")
    print("   2️⃣ If not found, searches the internet")
    print("   3️⃣ Only returns answer if confidence is 75% or higher")
    print("   4️⃣ Otherwise says 'check back later'")
    print("-" * 70)
    
    # Create agent
    try:
        agent = create_hybrid_agent("pdf")
        if not agent:
            print("\n⚠️ Agent created with internet search only (no PDFs found)")
        
        print("\n✅ Agent ready!")
        print("Type 'exit' to quit.")
        print("Type 'test' to run sample questions.")
        print("-" * 70)
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return
    
    # Test questions
    test_questions = [
        "Who is the CEO of Star-Grace IT Consulting?",
        "What services does Star-Grace IT Consulting offer?",
        "What is artificial intelligence?",
        "Who is the current President of Nigeria?",
        "What is the capital of France?",
        "How many employees does Star-Grace IT Consulting have?"
    ]
    
    conversation_history = []
    
    while True:
        user_input = input("\n❓ You: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'test':
            print("\n🧪 Running test questions...\n")
            print("=" * 70)
            for i, q in enumerate(test_questions, 1):
                print(f"\n❓ Q{i}: {q}")
                print("-" * 50)
                conversation_history.append(HumanMessage(content=q))
                result = agent.invoke({'messages': conversation_history})
                conversation_history = result['messages']
                time.sleep(1)  # Small delay between questions
            continue
        
        if not user_input:
            continue
        
        # Regular question
        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({'messages': conversation_history})
        conversation_history = result['messages']

if __name__ == "__main__":
    main()