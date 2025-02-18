import os
import streamlit as st
from dotenv import load_dotenv  # Load environment variables
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from urllib.parse import urlparse, urlunparse

# Load .env file
load_dotenv()

# Read API Key from .env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set it in the `.env` file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

VECTOR_STORE_PATH = "vector_store.faiss"

# Cache Vector Store
@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    return None

# Create Chatbot with Memory
@st.cache_resource
def create_rag_chain(_vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 20})  # Lowered k to reduce irrelevant sources
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        return_source_documents=True,  # Ensure sources are returned
        output_key="answer"  # Explicitly set the output key
    )

def normalize_url(url):
    parsed_url = urlparse(url)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))  # Ignore fragments

# Streamlit UI
def main():
    st.set_page_config(page_title="💬 AI-Powered RAG Chatbot", layout="wide")

    vector_store = load_vector_store()
    if not vector_store:
        st.warning("⚠️ No vector store found. Please ensure it is prepared before running the app!")
        return
    
    rag_chain = create_rag_chain(vector_store)

    st.title("💬 Chat with Your Website")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    query = st.text_input("🔍 Ask something:", key="query", on_change=lambda: st.session_state.update({'send_query': True}))
    
    if st.session_state.get("send_query") and query.strip():
        with st.spinner("🔎 Searching..."):
            try:
                result = rag_chain({"question": query, "chat_history": st.session_state["chat_history"]})
                
                answer = result.get("answer", "No answer found.")
                sources = result.get("source_documents", [])

                # Store chat history with sources
                st.session_state["chat_history"].insert(0, {
                    "role": "You",
                    "message": query
                })
                st.session_state["chat_history"].insert(0, {
                    "role": "Bot",
                    "message": answer,
                    "sources": sources  # Store sources with response
                })
                st.session_state.pop("send_query", None)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display Chat History with Sources
    for entry in st.session_state["chat_history"]:
        role = entry["role"]
        message = entry["message"]
        if role == "You":
            st.markdown(f"**🧑‍💻 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

            # Display sources for this response
            sources = entry.get("sources", [])
            if sources:
                st.subheader("📌 Sources for this response:")
                displayed_sources = 0
                unique_sources = set()
                for doc in sources:
                    source_url = doc.metadata.get('source', 'Unknown source')
                    normalized_source = normalize_url(source_url)
                    if normalized_source != 'Unknown' and normalized_source not in unique_sources:
                        unique_sources.add(normalized_source)
                        displayed_sources += 1
                        with st.expander(f"🔹 Source {displayed_sources}"):
                            st.write(f"**Source:** {source_url}")  
                        if displayed_sources == 5:
                            break  # Limit to top 5 unique sources
            else:
                st.write("_No sources found for this response._")

if __name__ == "__main__":
    main()
