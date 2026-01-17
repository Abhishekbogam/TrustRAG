import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"


st.markdown(
    """
    <style>
        /* Center main content */
        .block-container {
            max-width: 750px;
            margin: auto;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Chat input wrapper */
        section[data-testid="stChatInput"] {
            max-width: 750px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Actual input box */
        section[data-testid="stChatInput"] textarea {
            width: 100%;
            min-height: 90px;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)



import time

def type_text(container, text, delay=0.02):
    placeholder = container.empty()
    rendered = ""
    for char in text:
        rendered += char
        placeholder.markdown(rendered + "▌")
        time.sleep(delay)
    placeholder.markdown(rendered)


def build_conversation_memory(chat_history, max_turns=3):
    """
    Builds short conversational memory from last N turns.
    """
    memory_lines = []

    for role, message in chat_history[-(max_turns * 2):]:
        if role == "user":
            memory_lines.append(f"User: {message['text']}")
        elif role == "assistant" and message.get("summary"):
            memory_lines.append(f"Assistant: {message['summary']}")

    return "\n".join(memory_lines)


st.title("📄 RAG Application")


# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Upload Section
st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Ingest Documents") and uploaded_files:
    files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
    res = requests.post(f"{BACKEND_URL}/upload", files=files)

    if res.status_code == 200:
        st.success("Documents ingested successfully")
    else:
        st.error("Ingestion failed")


# Chat Rendering (ONLY FROM STATE)
st.subheader("Chat")

for role, message in st.session_state.chat_history:
    with st.chat_message(role):

        if role == "user":
            st.markdown(message["text"])

        elif role == "assistant":

            # Thinking state
            if message.get("status") == "thinking":
                st.markdown("⏳ Thinking...")

            else:
                # ---------------- Summary ----------------
                if message.get("summary"):
                    summary_container = st.container()

                    if not message.get("summary_animated"):
                        type_text(summary_container, message["summary"])
                        message["summary_animated"] = True
                    else:
                        summary_container.markdown(message["summary"])

                # ---------------- Bullet Points ----------------
                if message.get("bullet_points"):
                    bullets_container = st.container()

                    if not message.get("bullets_animated"):
                        for point in message["bullet_points"]:
                            type_text(bullets_container, f"- {point}", delay=0.01)
                        message["bullets_animated"] = True
                    else:
                        for point in message["bullet_points"]:
                            bullets_container.markdown(f"- {point}")

                # ---------------- Table (NO animation) ----------------
                table = message.get("table", {})
                if table.get("headers") and table.get("rows"):
                    st.table(
                        [dict(zip(table["headers"], row)) for row in table["rows"]]
                    )

                # ---------------- Sources ----------------
                if message.get("sources"):
                    st.markdown("**Sources:**")
                    for src in message["sources"]:
                        st.markdown(f"- {src}")



# Scroll Anchor 
st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)


# New Query Logic (TWO-PHASE RERUN)
user_query = st.chat_input("Ask a question...")

if user_query:
    #  Add user message
    st.session_state.chat_history.append(
        ("user", {"text": user_query})
    )

    #  Add assistant thinking placeholder
    st.session_state.chat_history.append(
        ("assistant", {"status": "thinking"})
    )

    # Anchor scroll BEFORE rerun
    st.markdown(
        """
        <script>
            document.getElementById("chat-bottom")?.scrollIntoView({behavior: "instant"});
        </script>
        """,
        unsafe_allow_html=True
    )

    # FIRST RERUN → show user + thinking
    st.rerun()


# Backend Call (ONLY WHEN THINKING)
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1][0] == "assistant"
    and st.session_state.chat_history[-1][1].get("status") == "thinking"
):
    last_user_message = None

    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            last_user_message = msg["text"]
            break

    memory = build_conversation_memory(st.session_state.chat_history)


    response = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": last_user_message,
            "memory":memory
            },
        timeout=60
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "summary": "Invalid response from backend.",
            "bullet_points": [],
            "table": {"headers": [], "rows": []}
        }

    # Replace thinking with final answer
    data["animated"] = False
    data["summary_animated"] = False
    data["bullets_animated"] = False
    st.session_state.chat_history[-1] = ("assistant", data)

    # Anchor scroll BEFORE rerun
    st.markdown(
        """
        <script>
            document.getElementById("chat-bottom")?.scrollIntoView({behavior: "instant"});
        </script>
        """,
        unsafe_allow_html=True
    )

    # SECOND RERUN → show final answer
    st.rerun()
