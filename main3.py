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
import random

# Load .env file
load_dotenv()

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
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 20})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    system_message = SystemMessage(content=(
        "Compare the chatbot's answer against the content of retrieved documents. "
        "Do not return information that is not found in the sources. Do not hallucinate. "
        "If the answer is not found, tell the user that the answer is not found on the Corporate Website."
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    enriched_prompt = llm.predict(f"In the context of Singapore Infocomm Media Development Authority, enrich this prompt for a more effective RAG search: {prompt}. Output only the prompt.")
    return enriched_prompt

def generate_follow_up_questions(answer):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = (
        f"Based on the following answer, suggest three follow-up questions that the user might ask. "
        f"Ensure the questions are relevant and engaging.\nAnswer: {answer}\nFollow-up questions:"
    )
    response = llm.predict(prompt)
    questions = response.split('\n')
    return [q.strip('- ') for q in questions if q.strip()][:3]

def main():
    st.set_page_config(page_title="💬 Ask CODI 2.0", layout="wide")

    vector_store = load_vector_store()
    if not vector_store:
        st.warning("⚠️ No vector store found. Please ensure it is prepared before running the app!")
        return

    rag_chain = create_rag_chain(vector_store)

    st.title("💬 Chat with CODI about Corporate Website")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "follow_up_questions" not in st.session_state:
        st.session_state["follow_up_questions"] = []

    query = st.session_state.pop("selected_question", "") or st.text_input("🔍 Ask something:", key="query_input")

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

                follow_up_questions = generate_follow_up_questions(answer)

                st.session_state["chat_history"].insert(0, {
                    "role": "You",
                    "message": query,
                    "answer": answer,
                    "sources": sources,
                    "follow_up_questions": follow_up_questions
                })
                st.session_state["follow_up_questions"] = follow_up_questions

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    for entry in st.session_state["chat_history"]:
        query = entry["message"]
        answer = entry["answer"]
        sources = entry.get("sources", [])
        follow_up_questions = entry.get("follow_up_questions", [])

        st.write(f"**🧑‍💻 You:** {query}")

        if follow_up_questions:
            st.markdown("**💡 Follow-up Questions:**")
            cols = st.columns(len(follow_up_questions))
            for idx, question in enumerate(follow_up_questions):
                if cols[idx].button(question, key=f"follow_up_{idx}_{random.randint(0, 100000)}"):
                    st.session_state.clear()
                    st.session_state["selected_question"] = question
                    st.rerun()
                    
        st.write(f"**🤖 Bot:** {answer}")

        if sources:
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
            st.write("_No sources found for this response._")

        st.markdown("---")

if __name__ == "__main__":
    main()
