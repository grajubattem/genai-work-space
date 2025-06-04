"""
Streamlit Chatbot UI using TogetherAI via LangChain + SQLite persistence
-----------------------------------------------------------------------
How to run:
    streamlit run streamlit_chatbot.py

Prerequisites:
    pip install streamlit langchain langchain-community python-dotenv
    # (TogetherAI dependencies are inside langchain-community)

Environment:
    Create a .env file in the same directory:
        TOGETHER_API_KEY=your_together_api_key_here
"""
import os
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Together

# ----------------- ENV & LLM SETUP -----------------
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

if not API_KEY:
    st.error("TOGETHER_API_KEY not found in environment. Add it to a .env file.")
    st.stop()

# Configure the TogetherAI LLM (Mistral‑7B‑Instruct by default)
@st.cache_resource(show_spinner=False)
def get_llm(model_id: str = "mistralai/Mistral-7B-Instruct-v0.1", temp: float = 0.7):
    return Together(
        model=model_id,
        temperature=temp,
        max_tokens=512,
        together_api_key=API_KEY,
    )

llm = get_llm()

# ----------------- SQLITE HELPER -----------------
class VzDatabase:
    def __init__(self, db_path: str = "vz1.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_schema()

    def _create_schema(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_query TEXT,
                llm_response TEXT
            )
            """
        )
        self.conn.commit()

    def insert(self, query: str, response: str):
        self.cursor.execute(
            "INSERT INTO chat_history (timestamp, user_query, llm_response) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(sep=" ", timespec="seconds"), query, response),
        )
        self.conn.commit()

    def fetch_all(self):
        self.cursor.execute("SELECT * FROM chat_history ORDER BY id DESC")
        return self.cursor.fetchall()

# Single DB instance for the app
@st.cache_resource(show_spinner=False)
def get_db():
    return VzDatabase()

db = get_db()

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="TogetherAI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 TogetherAI Chatbot")

with st.sidebar:
    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    model_id = st.selectbox(
        "Model",
        [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
    )
    if st.button("⚙️ Apply Changes"):
        llm = get_llm(model_id=model_id, temp=temperature)
        st.success("Model settings updated! New queries will use the updated model.")

# Session state for chat history display
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples (user, assistant)

# Input area
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_area("Your message:", placeholder="Ask me anything...", height=80)
    submitted = st.form_submit_button("Send")

if submitted and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            response_text = llm.invoke(user_query)
        except Exception as e:
            st.error(f"LLM invocation failed: {e}")
            response_text = "(Error generating response)"
    # Save and display
    st.session_state.history.append((user_query, response_text))
    db.insert(user_query, response_text)

# Display chat history
for user_msg, bot_msg in reversed(st.session_state.history):  # newest last
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# Optional expandable section to view DB contents
with st.expander("📜 Full conversation history (DB)"):
    rows = db.fetch_all()
    for _id, ts, q, r in rows:
        st.markdown(f"**[{ts}]** **You:** {q}")
        st.markdown(f"**Assistant:** {r}")
        st.markdown("---")
