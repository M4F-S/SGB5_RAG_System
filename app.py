"""
SGB V RAG System - Streamlit Web Interface
Interactive UI for querying the legal knowledge base
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="SGB V RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .section-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .source-box {
        background: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ SGB V Legal Knowledge Base</h1>
    <p>Retrieval-Augmented Generation for German Health Insurance Law (Sozialgesetzbuch V)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Check API health
try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✓ API Connected")
        api_health = response.json()
        st.sidebar.metric("Chunks Loaded", api_health.get("chunks_loaded", 0))
    else:
        st.sidebar.error("✗ API Error")
except:
    st.sidebar.error("✗ API Offline - Start server with: python -m src.api")

# Search type selection
search_type = st.sidebar.radio(
    "Search Type",
    options=["Semantic", "Lexical", "Hybrid"],
    help="Semantic: meaning-based | Lexical: keyword-based | Hybrid: both"
)

# Top K results
top_k = st.sidebar.slider("Number of Sources", min_value=1, max_value=10, value=5)

# Temperature for LLM
temperature = st.sidebar.slider(
    "Temperature (Lower = More Conservative)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Lower values for legal accuracy, higher for creativity"
)

# Example queries
st.sidebar.header("📚 Example Queries")
example_queries = [
    "What are the copayment requirements for prescriptions according to SGB V?",
    "Which sections cover the benefits for dental treatments?",
    "What is the maximum deductible (Selbstbeteiligung) for insured persons?",
    "How are chronic diseases (chronische Krankheiten) defined in SGB V?",
    "What are the rules for reimbursement of therapeutic aids?"
]

for i, query in enumerate(example_queries):
    if st.sidebar.button(f"Example {i+1}: {query[:30]}...", key=f"example_{i}"):
        st.session_state.selected_query = query

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📖 Browse Sections", "ℹ️ About"])

# ====================
# TAB 1: Query Interface
# ====================
with tab1:
    st.header("Query the SGB V Knowledge Base")
    
    # Query input
    if "selected_query" in st.session_state:
        query = st.text_area(
            "Enter your legal question:",
            value=st.session_state.selected_query,
            height=100,
            key="query_input"
        )
        del st.session_state.selected_query
    else:
        query = st.text_area(
            "Enter your legal question:",
            placeholder="e.g., What are the copayment rules for prescriptions?",
            height=100,
            key="query_input"
        )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit_button = st.button("🚀 Submit Query", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Process query
    if submit_button and query:
        if len(query.strip()) < 3:
            st.error("Please enter a question with at least 3 characters")
        else:
            with st.spinner("🔄 Processing query..."):
                try:
                    # Call API
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={
                            "question": query,
                            "top_k": top_k,
                            "temperature": temperature
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.success("✓ Answer generated successfully")
                        
                        st.markdown("### 📋 Answer")
                        st.markdown(f"> {result['answer']}")
                        
                        # Display sources
                        st.markdown("### 📚 Sources")
                        
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"[{i}] §{source['section_id']}: {source['title']}",
                                expanded=(i == 1)
                            ):
                                st.markdown(f"**Section:** §{source['section_id']}")
                                st.markdown(f"**Title:** {source['title']}")
                                st.markdown(f"**Relevance:** {source['relevance_score']:.1%}")
                                st.markdown("**Excerpt:**")
                                st.info(source['text'][:500] + "...")
                        
                        # Display metadata
                        with st.expander("📊 Metadata", expanded=False):
                            st.json(result['metadata'])
                        
                        # Copy to clipboard hint
                        st.info("💡 Tip: You can select and copy any text from the answer above")
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.json().get("detail", "Unknown error"))
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Try again with a shorter question.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif submit_button:
        st.warning("Please enter a question")

# ====================
# TAB 2: Browse Sections
# ====================
with tab2:
    st.header("📖 Browse SGB V Sections")
    
    # Search within sections
    section_search = st.text_input(
        "Search for sections:",
        placeholder="e.g., Beitrag, Leistungen, Krankengeld"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_in_sections = st.button("🔍 Search", use_container_width=True)
    with col2:
        list_all = st.button("📋 List All", use_container_width=True)
    
    if search_in_sections and section_search:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/search",
                    json={
                        "query": section_search,
                        "top_k": 10,
                        "search_type": search_type.lower()
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Found {results['total_results']} sections")
                    
                    for result in results['results']:
                        st.markdown(f"""
                        <div class="section-box">
                            <h4>§{result['section_id']}: {result['title']}</h4>
                            <p><small><strong>Category:</strong> {result['category']} | 
                            <strong>Relevance:</strong> {result['score']:.1%}</small></p>
                            <p>{result['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Search failed")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif list_all:
        with st.spinner("Loading sections..."):
            try:
                response = requests.get(f"{API_BASE_URL}/sections", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    st.info(f"Total Sections: {data['total_sections']}")
                    
                    # Create a table
                    import pandas as pd
                    df = pd.DataFrame(data['sections'])
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.error("Failed to load sections")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================
# TAB 3: About
# ====================
with tab3:
    st.header("ℹ️ About SGB V RAG System")
    
    st.markdown("""
    ### Overview
    This system provides intelligent retrieval of German Health Insurance Law (SGB V - Sozialgesetzbuch V)
    using advanced Retrieval-Augmented Generation (RAG) technology.
    
    ### How It Works
    
    1. **Retrieval**: Searches the complete SGB V database using:
       - Semantic search: Finds meaning-similar sections
       - Lexical search: Keyword-based search
       - Hybrid: Combines both methods
    
    2. **Generation**: Uses GPT-4 to generate accurate legal answers based on retrieved sections
    
    3. **Citation**: All answers include source citations with section numbers
    
    ### Key Features
    
    ✅ **German-Optimized**: Uses German-specific language models
    
    ✅ **Legally Accurate**: Temperature set to 0.3 for conservative, fact-based responses
    
    ✅ **Contextual**: Preserves document structure and cross-references
    
    ✅ **Cited**: Every answer includes section references
    
    ✅ **Fast**: FAISS vector database for quick semantic search
    
    ### Source Data
    - **Source**: gesetze-im-internet.de (official German legal repository)
    - **Coverage**: Complete SGB V (Health Insurance Code)
    - **Updated**: January 2026
    
    ### Legal Disclaimer
    ⚠️ **This system is for informational purposes only.**
    
    Answers generated by this system should not be considered legal advice.
    For official legal interpretation, always consult:
    - The official text at gesetze-im-internet.de
    - A qualified legal professional
    - Your Krankenkasse (health insurance provider)
    
    ### Technical Stack
    
    - **Framework**: LangChain + FastAPI + Streamlit
    - **Vector DB**: FAISS
    - **Embeddings**: sentence-transformers (Multilingual MPNet)
    - **LLM**: OpenAI GPT-4
    - **Language Processing**: spaCy + NLTK
    
    ### API Documentation
    
    View the interactive API docs at `/docs` when the server is running.
    
    ### Get Started
    
    1. Run the scraper: `python -m src.scraper`
    2. Process data: `python -m src.data_processor`
    3. Generate embeddings: `python -m src.embeddings`
    4. Start API: `python -m src.api`
    5. Launch UI: `streamlit run ui/app.py`
    """)
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p><strong>SGB V RAG System v1.0.0</strong></p>
        <p>Built with ❤️ for German legal document access</p>
        <p>© 2026 - Open Source Project</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📧 Questions? Check the documentation")
with col2:
    st.caption("📖 Learn more at: https://github.com/yourusername/sgbv_rag_system")
with col3:
    st.caption(f"⏰ Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
