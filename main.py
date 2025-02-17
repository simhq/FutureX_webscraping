import os
import streamlit as st
from dotenv import load_dotenv  # Load environment variables
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# Create Chatbot with Memory
@st.cache_resource
def create_rag_chain(_vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)  # Use ChatOpenAI instead of OpenAI
    retriever = _vector_store.as_retriever(search_kwargs={"k": 20})  
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

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

                # Store chat history (latest first)
                st.session_state["chat_history"] = [("You", query), ("Bot", answer)] + st.session_state["chat_history"]
                st.session_state.pop("send_query", None)

                # Display Chat History
                for role, message in st.session_state["chat_history"]:
                    if role == "You":
                        st.markdown(f"**🧑‍💻 You:** {message}")
                    else:
                        st.markdown(f"**🤖 Bot:** {message}")

                # Display Sources
                st.subheader("📌 Sources:")
                if sources:
                    for i, doc in enumerate(sources):
                        with st.expander(f"🔹 Source {i+1}"):
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                st.markdown(f"[🔗 Source {i+1}]({doc.metadata['source']})")
                            else:
                                st.write(doc.page_content)
                else:
                    st.write("No sources found.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
