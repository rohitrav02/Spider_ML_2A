import streamlit as st
import pandas as pd
import plotly.express as px
from rag import RAGSystem, BENCHMARK_QUESTIONS

# Page config
st.set_page_config(
    page_title="RAG System - Research Papers",
    layout="wide"
)

st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
.main {
    font-family: 'Inter', sans-serif;
}

/* Main title */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 30px;
    letter-spacing: -0.02em;
}

/* Answer box with gradient border */
.answer-box {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    color: #1a202c;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin: 15px 0;
    position: relative;
    overflow: hidden;
}

.answer-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Retrieval box with subtle styling */
.retrieval-box {
    background: linear-gradient(145deg, #f7fafc 0%, #edf2f7 100%);
    color: #2d3748;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.9em;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
}

.retrieval-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.1);
}

/* Success message styling */
.success-message {
    background: linear-gradient(135deg, #68d391 0%, #38a169 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 500;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Info cards */
.info-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Section headers */
.section-header {
    color: #ffffff;
    font-weight: 600;
    font-size: 1.3em;
    margin-bottom: 15px;
    padding-bottom: 8px;
}

/* Benchmark button styling */
.benchmark-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    margin: 4px 0;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
}

.benchmark-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

/* Footer styling */
.footer {
    margin-top: 40px;
    padding: 20px;
    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    border-radius: 8px;
    text-align: center;
    color: #4a5568;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Loading spinner custom */
.loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

/* Status indicators */
.status-success {
    color: #38a169;
    font-weight: 600;
}

.status-error {
    color: #e53e3e;
    font-weight: 600;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system."""
    rag = RAGSystem()
    rag.initialize()
    return rag

def main():
    st.markdown('<h1 class="main-title"> Research Paper RAG System</h1>', unsafe_allow_html=True)
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'process_question' not in st.session_state:
        st.session_state.process_question = False
    
    with st.spinner(" Loading RAG system..."):
        rag_system = load_rag_system()
    
    st.markdown('<div class="success-message"> RAG system loaded successfully!</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header"> Ask Questions</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs([" Custom Question", " Benchmark Questions"])
        
        with tab1:
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="What would you like to know about the research papers?",
                value=st.session_state.current_question if st.session_state.current_question else ""
            )
            
            if st.button(" Ask Question", type="primary", use_container_width=True):
                st.session_state.current_question = question
                st.session_state.process_question = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.write("**Select a benchmark question:**")
            
            for i, q in enumerate(BENCHMARK_QUESTIONS):
                question_preview = q if len(q) <= 80 else q[:80] + "..."
                if st.button(f"Q{i+1}: {question_preview}", key=f"bench_{i}", use_container_width=True):
                    st.session_state.current_question = q
                    st.session_state.process_question = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header"> System Info</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2em; font-weight: 700;">{len(rag_system.documents)}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">Papers Loaded</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(" Run All Benchmarks", type="secondary", use_container_width=True):
            st.session_state.run_benchmarks = True

    if st.session_state.process_question and st.session_state.current_question.strip():
        with st.spinner(" Generating answer..."):
            result = rag_system.query(st.session_state.current_question)
            
            st.markdown('<div class="section-header"> Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-header"> Retrieval Details</div>', unsafe_allow_html=True)
            
            if result["retrieval_info"]:
                df = pd.DataFrame(result["retrieval_info"])
                fig = px.bar(
                    df, 
                    x='chunk_id', 
                    y='similarity_score',
                    color='source',
                    title="Similarity Scores by Retrieved Chunks",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", size=12),
                    title_font=dict(size=16, color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                for info in result["retrieval_info"]:
                    st.markdown(f"""
                    <div class="retrieval-box">
                        <strong> Chunk {info['chunk_id']}</strong> from <em>{info['source']}</em> 
                        <span style="color: #667eea; font-weight: 600;">(Similarity: {info['similarity_score']:.4f})</span><br>
                        <div style="margin-top: 8px; font-size: 0.85em; color: #4a5568;">{info['preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.session_state.process_question = False

    if st.session_state.get('run_benchmarks', False):
        st.markdown('<div class="section-header"> Benchmark Results</div>', unsafe_allow_html=True)
        
        results = []
        progress = st.progress(0)
        status_container = st.empty()
        
        for i, bench_q in enumerate(BENCHMARK_QUESTIONS):
            progress.progress((i + 1) / len(BENCHMARK_QUESTIONS))
            status_container.write(f"Processing question {i+1}/{len(BENCHMARK_QUESTIONS)}...")
            
            try:
                result = rag_system.query(bench_q)
                results.append({
                    "Question": f"Q{i+1}",
                    "Answer Preview": result["answer"][:100] + "...",
                    "Status": " Success"
                })
            except Exception as e:
                results.append({
                    "Question": f"Q{i+1}",
                    "Answer Preview": f"Error: {str(e)}",
                    "Status": " Failed"
                })
        
        status_container.empty()
        
        df_results = pd.DataFrame(results)
        st.dataframe(
            df_results, 
            use_container_width=True,
            hide_index=True
        )
        st.session_state.run_benchmarks = False

    st.markdown("""
    <div class="footer">
        <strong> Research Papers:</strong> Attention Is All You Need • BERT • GPT-3 • CLIP • LLaMA
        <br>
        <small style="opacity: 0.7;">Powered by RAG Technology</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()