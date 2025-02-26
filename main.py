import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from urllib.parse import urlparse, urlunparse

# Load .env file
load_dotenv()

# Custom CSS for styling
custom_css = """
<style>
    .pilot-banner {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .ai-response {
        background-color: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .verification-reminder {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
"""

# Read API Key from .env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set it in the `.env` file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

VECTOR_STORE_PATH = "vector_store.faiss"

@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    return None

@st.cache_resource
def create_rag_chain(_vector_store):
    llm = ChatOpenAI(model="gpt-4-mini", temperature=0)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    system_message = SystemMessage(content=(
        "Strictly use the retrieved context verbatim. Do not generate information beyond the provided context. Do not hallucinate."
        "If the answer is not found, tell the user that the answer is not found on the Corporate Website."
        "Ensure consistent font usage throughout the response."
    ))
    memory.chat_memory.add_message(system_message)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        return_source_documents=True,
        output_key="answer"
    )

def normalize_url(url):
    parsed_url = urlparse(url)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))

def enhance_prompt(prompt):
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    enriched_prompt = llm.predict(f"In the context of Singapore Infocomm Media Development Authority, enrich this prompt for a more effective RAG search. If the prompt is a name, the enriched prompt shall include the possibility that this person is part of the senior management, or this person is part of the management in a Group in IMDA: {prompt}. Output only the prompt.")
    return enriched_prompt

def main():
    st.set_page_config(page_title="💬 Ask CODI 2.0", layout="wide")
    
    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Add pilot warning banner
    st.markdown("""
        <div class="pilot-banner">
            <h3>🔬 PILOT SYSTEM</h3>
            <p>This is an experimental AI-powered assistant currently in pilot phase. 
            Responses may not be fully accurate or complete.</p>
        </div>
    """, unsafe_allow_html=True)

    # Add informational sidebar
    with st.sidebar:
        st.header("ℹ️ About CODI 2.0")
        st.markdown("""
        **Important Information:**
        - This is a pilot system using Large Language Models (LLMs)
        - All responses are AI-generated and may contain errors
        - Always verify important information with official sources
        - Use specific, clear questions for better results
         """)
        
    vector_store = load_vector_store()
    if not vector_store:
        st.warning("⚠️ No vector store found. Please ensure it is prepared before running the app!")
        return

    rag_chain = create_rag_chain(vector_store)

    st.title("💬 Ask CODI 2.0")
    
    # Add AI-generated content disclaimer
    st.caption("All responses are AI-generated and should be verified against official sources")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "query" not in st.session_state:
        st.session_state["query"] = ""

    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input(
            "🔍 Ask CODI:", 
            key="query",
            help="Enter your question. Remember to verify any critical information.",
            placeholder="Type your question here..."
        )
    with col2:
        if st.button("🆕 New Query"):
            st.session_state["chat_history"] = []
            st.session_state["query"] = ""
            rag_chain.memory.clear()
            st.rerun()

    if query.strip():
        with st.spinner("🔎 Searching..."):
            try:
                enhanced_query = enhance_prompt(query)

                chat_history = []
                for entry in st.session_state["chat_history"]:
                    if entry["role"] == "You":
                        chat_history.append(HumanMessage(content=entry["message"]))
                    else:
                        chat_history.append(AIMessage(content=entry["answer"]))

                result = rag_chain({"question": enhanced_query, "chat_history": chat_history})

                answer = result.get("answer", "No answer found on the Corporate Website.")
                sources = result.get("source_documents", [])

                if not sources:
                    answer = "No answer found on the Corporate Website."

                st.session_state["chat_history"].insert(0, {
                    "role": "You",
                    "message": query,
                    "answer": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    for entry in st.session_state["chat_history"]:
        query = entry["message"]
        answer = entry["answer"]
        sources = entry.get("sources", [])

        st.write(f"**🧑‍💻 You:** {query}")
        
        # AI response container with verification reminder
        st.markdown("""
            <div class="ai-response">
                <strong>🤖 AI Response:</strong><br>
                {answer}
                <div class="verification-reminder">
                    ⚠️ This is an AI-generated response. Please verify critical information.
                </div>
            </div>
        """.format(answer=answer), unsafe_allow_html=True)

        if sources:
            st.markdown("**📚 Verification Sources:**")
            unique_sources = set()
            displayed_sources = 0
            for doc in sources:
                source_url = doc.metadata.get('source', 'Unknown source')
                normalized_source = normalize_url(source_url).lower()
                if normalized_source != 'unknown' and normalized_source not in unique_sources:
                    unique_sources.add(normalized_source)
                    displayed_sources += 1
                    st.write(f"**📌 Source {displayed_sources}:** {source_url}")
                    if displayed_sources == 5:
                        break
        else:
            st.warning("_No sources found for this response. Extra verification is recommended._")

        st.markdown("---")

    # Footer with additional reminders
    st.markdown("""
        <div style='position: fixed; bottom: 0; left: 0; right: 0; background-color: #f8f9fa; padding: 10px; text-align: center; font-size: 0.8rem;'>
            🔍 Remember: Always verify AI-generated responses against official sources
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()