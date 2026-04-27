# Streamlit UI and session management
# Features:
#   - Persistent per-user chat history (survives logout/restart)
#   - Signup page: new users register and are written to Postgres 
#   - Guest Chat → Personal Account transfer on login
#   - Sidebar with named past chats
#   - Rename past chats inline
#   - Thread ID updates to show the active/viewed chat's thread
#   - Preview/expander for every memory layer
#   - Guest mode with Buffer Window Memory only
#   - Strict per-user memory isolation (user_id column in every DB table)

import streamlit as st
import os
import uuid
import streamlit_authenticator as stauth
from rag_system import RAGEngine
from chat_store import (
    load_chat_history,
    save_chat_history,
    save_guest_session,
    load_guest_session,
    delete_guest_session,
    transfer_guest_to_user,
)
import user_db  # PostgreSQL user management

st.set_page_config(page_title="RAG Assistant", layout="wide")

# ── Sequential chat name generator ───────────────────────────────────────────
def _next_chat_name() -> str:
    """Return sequential name: Chat 1, Chat 2, … based on history length."""
    n = len(st.session_state.get("chat_history", [])) + 1
    return f"Chat {n}"


# ── DB bootstrap ──────────────────────────────────────────────────────────────
# Ensure PostgreSQL schema exists on first run.
user_db.init_db()


def _build_auth_config() -> dict:
    """
    Build the streamlit-authenticator config dict from Postgres.
    PostgreSQL is the single source of truth for all user accounts.
    """
    config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "expiry_days": 30,
            "key": "rag_app_secret_key",
            "name": "rag_cookie",
        },
    }

    db_users = user_db.get_all_users_for_auth_yaml()
    for u in db_users:
        config["credentials"]["usernames"][u["user_id"]] = {
            "email":    u["email"],
            "name":     u["display_name"],
            "password": u["password_hash"],
        }
    return config


# ── Signup page ───────────────────────────────────────────────────────────────
def _show_signup_page() -> None:
    """
    Render the self-service registration form.

    On success:
      • Inserts the new user into Postgres (users table, user_id column).
      • Flips st.session_state.page back to 'login' so the user can sign in.
    """
    st.title("Create an Account")
    st.caption("Fill in the details below to register.")

    with st.form("signup_form", clear_on_submit=False):
        new_username    = st.text_input("Username", placeholder="e.g. alice123")
        new_name        = st.text_input("Display Name", placeholder="e.g. Alice")
        new_email       = st.text_input("Email", placeholder="alice@example.com")
        new_password    = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted       = st.form_submit_button("Register", use_container_width=True)

    if submitted:
        # ── Validation ────────────────────────────────────────────────────────
        errors = []
        new_username = new_username.strip().lower()
        new_email    = new_email.strip().lower()

        if not new_username:
            errors.append("Username is required.")
        elif len(new_username) < 3:
            errors.append("Username must be at least 3 characters.")
        elif not new_username.isalnum():
            errors.append("Username may only contain letters and numbers.")

        if not new_name.strip():
            errors.append("Display name is required.")
        if not new_email or "@" not in new_email:
            errors.append("A valid email address is required.")
        if len(new_password) < 6:
            errors.append("Password must be at least 6 characters.")
        if new_password != confirm_password:
            errors.append("Passwords do not match.")

        if not errors:
            if user_db.user_exists(new_username):
                errors.append(f"Username '{new_username}' is already taken.")
            if user_db.email_exists(new_email):
                errors.append(f"Email '{new_email}' is already registered.")

        if errors:
            for e in errors:
                st.error(e)
            return

        # ── Persist to Postgres ───────────────────────────────────────────────
        hashed = stauth.Hasher.hash(new_password)
        ok = user_db.create_user(
            user_id=new_username,
            email=new_email,
            display_name=new_name.strip(),
            password_hash=hashed,
        )

        if ok:
            st.success(
                f"✅ Account created for **{new_name.strip()}** (`{new_username}`)! "
                "You can now log in."
            )
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Registration failed. Please try again.")

    st.divider()
    if st.button("← Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# ── Page routing ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "signup":
    _show_signup_page()
    st.stop()

# ── Authentication (login page) ───────────────────────────────────────────────
config = _build_auth_config()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login()


# Shared engine (created once, shared between guest and auth) 
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
engine = st.session_state.engine


#  AUTHENTICATED USER
if st.session_state["authentication_status"]:

    user_id: str = st.session_state.get("username") or "default"

    # Stamp last_login in Postgres for this user
    user_db.update_last_login(user_id)

    # Load persistent chat history from disk on first run 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(user_id)

    # If user changed (different account), reload their history AND their chunks
    if st.session_state.get("_loaded_user_id") != user_id:
        st.session_state.chat_history = load_chat_history(user_id)
        st.session_state["_loaded_user_id"] = user_id
        # Restore only this user's previously-indexed documents
        engine.load_session_data(user_id)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "viewing_thread_id" not in st.session_state:
        st.session_state.viewing_thread_id = None

    if "renaming_chat_idx" not in st.session_state:
        st.session_state.renaming_chat_idx = None

    
    #  GUEST CHAT TRANSFER — runs once on login if guest data exists
    if not st.session_state.get("_guest_transfer_done"):
        # Check if there's a guest session in session_state from before login
        guest_tid  = st.session_state.get("guest_thread_id")
        guest_msgs = st.session_state.get("guest_messages", [])

        transferred = False
        if guest_tid and guest_msgs:
            # Transfer: reassign guest chat ownership to this user account
            transferred = transfer_guest_to_user(
                guest_thread_id=guest_tid,
                guest_messages=guest_msgs,
                user_id=user_id,
            )
            if transferred:
                # Reload history from disk (now includes the transferred chat)
                st.session_state.chat_history = load_chat_history(user_id)
                # Also transfer the buffer window memory from guest to user
                try:
                    engine.memory_mgr.clear_guest_session(guest_tid)
                except Exception:
                    pass

        # Always purge any guest-indexed chunks from ChromaDB so the logged-in
        # user starts with only their own previously-indexed documents.
        if guest_tid:
            engine.clear_guest_data(guest_tid)

        # Clean up guest session state regardless
        for key in ("guest_thread_id", "guest_messages",
                    "_guest_pending_pdf_name", "_guest_pending_pdf_bytes"):
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["_guest_transfer_done"] = True

        if transferred:
            st.toast("✅ Your guest chat has been transferred to your account!", icon="🔄")

    # Helper: save current chat to history + persist to disk 
    def _archive_current_chat() -> bool:
        """Push the current live chat into chat_history with a sequential name.
        Returns True if archived, False if the chat was blank."""
        msgs = st.session_state.messages
        if not msgs:
            return False
        name = _next_chat_name()
        tid  = st.session_state.thread_id
        st.session_state.chat_history.insert(0, {
            "thread_id": tid,
            "name":      name,
            "messages":  list(msgs),
        })
        save_chat_history(user_id, st.session_state.chat_history)
        # Also record in Postgres per-user conversation index
        user_db.save_conversation(user_id, tid, title=name)
        return True

    def _save_history() -> None:
        """Save the current chat_history to disk."""
        save_chat_history(user_id, st.session_state.chat_history)

    # Compute displayed thread ID 
    viewing_id = st.session_state.viewing_thread_id
    if viewing_id is not None:
        display_thread_id = viewing_id
    else:
        display_thread_id = st.session_state.thread_id

    # Sidebar 
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout('Logout', 'main')
        st.divider()

        # New Chat button (always enabled) 
        if st.button(
            "＋  New Chat",
            use_container_width=True,
            type="primary",
        ):
            has_messages = bool(st.session_state.messages)

            # Only archive + summarise if the live chat actually has messages
            if has_messages:
                _archive_current_chat()

                try:
                    new_tid = engine.new_chat_thread(
                        user_id   = user_id,
                        is_guest  = False,
                        thread_id = st.session_state.thread_id,
                    )
                except Exception as exc:
                    print(f"[New Chat] Summarisation failed (non-blocking): {exc}")
                    new_tid = str(uuid.uuid4())

                st.session_state.thread_id = new_tid
                st.session_state.messages  = []

            # Always return to live chat view (handles: viewing past chat → new chat)
            st.session_state.viewing_thread_id = None
            st.session_state.renaming_chat_idx = None
            st.rerun()

        # Past chats list 
        if st.session_state.chat_history:
            st.markdown("#### 💬 Past Chats")
            for idx, chat in enumerate(st.session_state.chat_history):
                is_viewing = (
                    st.session_state.viewing_thread_id == chat["thread_id"]
                )
                is_renaming = (st.session_state.renaming_chat_idx == idx)
                tid_short = chat["thread_id"][-8:]

                if is_renaming:
                    new_name = st.text_input(
                        "Rename chat",
                        value=chat["name"],
                        key=f"rename_input_{idx}_{tid_short}",
                        label_visibility="collapsed",
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("✓ Save", key=f"save_rename_{idx}_{tid_short}",
                                     use_container_width=True):
                            if new_name.strip():
                                st.session_state.chat_history[idx]["name"] = new_name.strip()
                                _save_history()
                            st.session_state.renaming_chat_idx = None
                            st.rerun()
                    with col_cancel:
                        if st.button("✕ Cancel", key=f"cancel_rename_{idx}_{tid_short}",
                                     use_container_width=True):
                            st.session_state.renaming_chat_idx = None
                            st.rerun()
                else:
                    label = f"{'▶ ' if is_viewing else ''}{chat['name']}"

                    col_btn, col_ren, col_del = st.columns([5, 1, 1])
                    with col_btn:
                        if st.button(label, key=f"chat_{idx}_{tid_short}",
                                     use_container_width=True):
                            st.session_state.viewing_thread_id = chat["thread_id"]
                            st.session_state.renaming_chat_idx = None
                            st.rerun()
                    with col_ren:
                        if st.button("✏️", key=f"ren_{idx}_{tid_short}",
                                     help=f"Rename {chat['name']}"):
                            st.session_state.renaming_chat_idx = idx
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{idx}_{tid_short}",
                                     help=f"Delete {chat['name']}"):
                            if st.session_state.viewing_thread_id == chat["thread_id"]:
                                st.session_state.viewing_thread_id = None
                            st.session_state.chat_history.pop(idx)
                            st.session_state.renaming_chat_idx = None
                            _save_history()
                            st.rerun()

            if st.session_state.viewing_thread_id is not None:
                st.divider()
                if st.button("↩ Back to Current Chat", use_container_width=True):
                    st.session_state.viewing_thread_id = None
                    st.rerun()

        st.divider()
        st.caption(f"Thread: `{display_thread_id}`")
        st.divider()

        
        #  MEMORY LAYERS with previews
        st.header("Memory Layers")
        diag = engine.memory_diagnostics(user_id)

        st.markdown("**Per-Conversation Memory :**")

        n_turns = diag.get("session_turns", 0)
        st.markdown(f"-> **Buffer Window** — {n_turns} turn{'s' if n_turns != 1 else ''}")
        if n_turns > 0:
            with st.expander("Preview Buffer Window", expanded=False):
                pairs = engine.memory_mgr.get_buffer_window_pairs(user_id)
                for i, (h, a) in enumerate(pairs, 1):
                    st.caption(f"**Turn {i}**")
                    st.markdown(f"  🧑 {h[:150]}{'…' if len(h) > 150 else ''}")
                    st.markdown(f"  🤖 {a[:150]}{'…' if len(a) > 150 else ''}")

        n_facts = diag.get("user_memory_facts", 0)
        st.markdown(f"-> **User Memory** — {n_facts} fact{'s' if n_facts != 1 else ''}")
        if n_facts > 0:
            with st.expander("Manage Memories", expanded=False):
                for mem in engine.memory_mgr.list_memories(user_id):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        tag = "💬" if mem["source"] == "explicit" else "🤖"
                        st.caption(f"{tag} {mem['text']}")
                    with col2:
                        if st.button("✕", key=f"forget_{mem['id']}"):
                            engine.memory_mgr.forget(mem["id"], user_id)
                            st.rerun()

        n_convs = diag.get("recent_convs_stored", 0)
        has_active = diag.get("has_active_session", False)
        archived = diag.get("archived_convs", 0)
        if has_active and archived > 0:
            label = f"1 active + {archived} archived"
        elif has_active:
            label = "1 active"
        elif archived > 0:
            label = f"{archived} archived"
        else:
            label = "0 conversations"
        st.markdown(f"-> **Summary Memory** — {label}")
        if n_convs > 0:
            with st.expander("Preview Summaries", expanded=False):
                summaries = engine.memory_mgr.get_recent_conversation_summaries(user_id)
                for i, s in enumerate(summaries, 1):
                    is_live = s.get("thread_id") == "__live__"
                    tag = " ACTIVE" if is_live else ""
                    st.caption(f"**{i}. {s['date']}: \"{s['title']}\"** {tag}")
                    for b in s["bullets"]:
                        st.markdown(f"  — {b}")

        kg = diag.get("kg_stats", {})
        n_kg_edges = kg.get("edges", 0)
        st.markdown(f"-> **User KG** — {kg.get('nodes', 0)} nodes / {n_kg_edges} edges")
        if n_kg_edges > 0:
            with st.expander("Preview Knowledge Graph", expanded=False):
                edges = engine.memory_mgr.get_kg_edges(user_id)
                for e in edges[:20]:
                    st.caption(f"**{e['subject']}** —[ {e['relation']} ]→ **{e['object']}**")
                if len(edges) > 20:
                    st.caption(f"… and {len(edges) - 20} more triples")

        vs = diag.get("vector_store_stats", {})
        st.markdown(f"-> **Vector Store** — {vs.get('stored_memories', 0)} stored memories")

        st.markdown("")
        st.markdown(" **Global + Account Memory :**")

        n_global_facts = diag.get("global_summary_facts", 0)
        st.markdown(f"-> **Global Summary** — {n_global_facts} fact{'s' if n_global_facts != 1 else ''}")
        if n_global_facts > 0:
            with st.expander("Preview Global Facts", expanded=False):
                gfacts = engine.memory_mgr.get_global_facts()
                for gf in gfacts:
                    st.caption(f"• {gf['text']}  _(from {gf['source_user']})_")

        gkg = diag.get("global_kg_stats", {})
        n_gkg_edges = gkg.get("edges", 0)
        st.markdown(f"-> **Account KG** — {gkg.get('nodes', 0)} nodes / {n_gkg_edges} edges")
        if n_gkg_edges > 0:
            with st.expander("Preview Account KG", expanded=False):
                gedges = engine.memory_mgr.get_global_kg_edges(user_id)
                for e in gedges[:20]:
                    st.caption(f"**{e['subject']}** —[ {e['relation']} ]→ **{e['object']}**")
                if len(gedges) > 20:
                    st.caption(f"… and {len(gedges) - 20} more triples")

        st.markdown("")
        st.markdown(" **Session Metadata** — active")
        with st.expander("Preview Session Metadata", expanded=False):
            smeta = engine.memory_mgr.get_session_metadata(user_id)
            st.caption(f"• User: {smeta['user_id']}")
            st.caption(f"• Session ID: {smeta['session_id']}")
            st.caption(f"• Tier: {smeta['subscription_tier']}")
            st.caption(f"• Started: {smeta['session_start']}")
            st.caption(f"• Total conversations: {smeta['total_conversations']}")
            st.caption(f"• Total messages: {smeta['total_messages']}")
            st.caption(f"• Avg msgs/conv: {smeta['avg_messages_per_conv']:.1f}")

        st.divider()

        # Settings 
        st.header("Settings")

        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            st.session_state["_pending_pdf_name"]  = uploaded_file.name
            st.session_state["_pending_pdf_bytes"] = uploaded_file.getvalue()

        pending_name  = st.session_state.get("_pending_pdf_name")
        pending_bytes = st.session_state.get("_pending_pdf_bytes")

        if pending_name and pending_bytes:
            if st.button(f"Index  ›  {pending_name}", use_container_width=True):
                with st.spinner("Processing…"):
                    temp_path = f"temp_{pending_name}"
                    with open(temp_path, "wb") as f:
                        f.write(pending_bytes)
                    engine.process_pdf(
                        temp_path,
                        doc_name=pending_name,
                        session_id=user_id,   # tag chunks with the user's id
                    )
                    os.remove(temp_path)
                    del st.session_state["_pending_pdf_name"]
                    del st.session_state["_pending_pdf_bytes"]
                    st.success(f"✅ '{pending_name}' indexed — {len(engine._chunks)} chunks loaded!")
                    st.rerun()
        else:
            st.button("Index Document", disabled=True, use_container_width=True)

        st.divider()

        if st.button("Clear Collection", type="secondary"):
            if engine.clear_all_data(user_id=user_id):
                st.session_state.messages = []
                st.success("Collection cleared!")
                st.rerun()

    # Main chat area 

    if viewing_id is not None:
        past = next(
            (c for c in st.session_state.chat_history if c["thread_id"] == viewing_id),
            None,
        )
        if past:
            st.title(f"📖 {past['name']}")
            st.caption("This is a read-only view of a past conversation.")
            st.divider()
            for message in past["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.warning("Could not find this past chat.")

    else:
        st.title("RAG Assistant")

        if engine._chunks:
            st.success(f"✅ Document ready — {len(engine._chunks)} chunks indexed", icon=None)
        else:
            st.info("ℹ️ No document indexed yet. You can still chat, or upload a PDF for document-based Q&A.", icon=None)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything..."):

            remember_match = (
                prompt.lower().startswith("remember that ") or
                prompt.lower().startswith("remember: ")
            )
            forget_match = (
                prompt.lower().startswith("forget ") or
                prompt.lower().startswith("forget that ")
            )

            if remember_match:
                fact_text = prompt.split(" ", maxsplit=2)[-1].strip()
                engine.memory_mgr.remember(fact_text, user_id)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    msg = f"I'll remember that: *{fact_text}*"
                    st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()

            elif forget_match:
                forget_text = prompt.split(" ", maxsplit=1)[-1].strip()
                removed = engine.memory_mgr.forget(forget_text, user_id)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    msg = (
                        f"Memory removed: *{forget_text}*"
                        if removed else
                        f"I couldn't find a memory matching: *{forget_text}*"
                    )
                    st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()

            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = engine.chat(
                            query     = prompt,
                            thread_id = st.session_state.thread_id,
                            user_id   = user_id,
                            is_guest  = False,
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                        })


#  GUEST MODE (not authenticated — Buffer Window Memory only)
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:

    st.warning('Please enter your username and password')

    # ── Sign Up link ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 1])
    with col_r:
        if st.button("Create Account", use_container_width=True, type="secondary"):
            st.session_state.page = "signup"
            st.rerun()

    st.divider()

    st.subheader("💬 Guest Chat (limited memory)")
    st.caption(
        "You are chatting as a guest. Only **Buffer Window Memory** (short-term context) "
        "is available. Log in for full memory capabilities including Knowledge Graph, "
        "Summary Memory, and Vector Store Retriever."
    )
    st.caption("💡 **Tip:** Your guest chat will be **automatically transferred** to your account when you log in.")

    # Guest session init 
    if "guest_thread_id" not in st.session_state:
        new_guest_id = f"guest_{uuid.uuid4().hex[:8]}"
        st.session_state.guest_thread_id = new_guest_id
        # Clear any previously-indexed guest documents from the engine so
        # this visitor starts with a clean slate (no cross-session data leakage).
        engine.clear_guest_data(new_guest_id)  # no-op for brand-new ids
        engine._chunks        = []             # ensure in-memory cache is empty
        engine._docs_embed    = __import__("numpy").empty((0,))
        engine._bm25_index    = None
        engine._metadata      = []
        engine._uploaded_docs = []
    if "guest_messages" not in st.session_state:
        st.session_state.guest_messages = []

    # Reset transfer flag so it runs again after next login
    st.session_state["_guest_transfer_done"] = False

    if engine._chunks:
        st.success(f"✅ Document ready — {len(engine._chunks)} chunks indexed", icon=None)
    else:
        st.info("ℹ️ No document indexed. You can still chat, or log in to upload a PDF.", icon=None)

    for message in st.session_state.guest_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything (guest mode)..."):
        st.session_state.guest_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = engine.chat(
                    query     = prompt,
                    thread_id = st.session_state.guest_thread_id,
                    user_id   = None,
                    is_guest  = True,
                )
                st.markdown(response)
                st.session_state.guest_messages.append({
                    "role": "assistant",
                    "content": response,
                })

                # Persist guest session to disk after each message
                save_guest_session(
                    st.session_state.guest_thread_id,
                    st.session_state.guest_messages,
                )

    # Guest sidebar 
    with st.sidebar:
        st.markdown("### Guest Mode")
        st.markdown("🔒 **Limited Memory**")
        st.markdown("• Buffer Window Memory only")
        st.markdown("• No persistent memory")
        st.markdown("• No Knowledge Graph")
        st.markdown("• No Vector Store")
        st.divider()
        n_guest_msgs = len(st.session_state.guest_messages)
        if n_guest_msgs > 0:
            st.caption(f"💬 {n_guest_msgs} message{'s' if n_guest_msgs != 1 else ''} in guest session")
            st.caption("These will transfer to your account on login.")
        st.divider()

        # Guest PDF upload — chunks are tagged with guest_thread_id and
        # automatically purged when this session ends or the user logs in.
        st.subheader("Upload PDF (Guest)")
        guest_pdf = st.file_uploader("Upload PDF", type="pdf", key="guest_pdf_uploader")
        if guest_pdf is not None:
            st.session_state["_guest_pending_pdf_name"]  = guest_pdf.name
            st.session_state["_guest_pending_pdf_bytes"] = guest_pdf.getvalue()

        g_pending_name  = st.session_state.get("_guest_pending_pdf_name")
        g_pending_bytes = st.session_state.get("_guest_pending_pdf_bytes")
        if g_pending_name and g_pending_bytes:
            if st.button(f"Index  ›  {g_pending_name}", use_container_width=True, key="guest_index_btn"):
                with st.spinner("Processing…"):
                    temp_path = f"temp_guest_{g_pending_name}"
                    with open(temp_path, "wb") as f:
                        f.write(g_pending_bytes)
                    engine.process_pdf(
                        temp_path,
                        doc_name=g_pending_name,
                        session_id=st.session_state.guest_thread_id,  # guest-scoped
                    )
                    os.remove(temp_path)
                    del st.session_state["_guest_pending_pdf_name"]
                    del st.session_state["_guest_pending_pdf_bytes"]
                    st.success(f"✅ '{g_pending_name}' indexed!")
                    st.rerun()
        else:
            st.button("Index Document", disabled=True, use_container_width=True, key="guest_index_disabled")

        st.divider()

        # End Session: purge guest chunks from ChromaDB and reset state
        if st.button("🗑 End Guest Session", use_container_width=True, type="secondary"):
            removed = engine.clear_guest_data(st.session_state.guest_thread_id)
            delete_guest_session(st.session_state.guest_thread_id)
            for key in ("guest_thread_id", "guest_messages",
                        "_guest_pending_pdf_name", "_guest_pending_pdf_bytes"):
                st.session_state.pop(key, None)
            if removed:
                st.success(f"Session ended — {removed} indexed chunk(s) removed.")
            else:
                st.success("Session ended.")
            st.rerun()

        st.divider()
        st.caption(f"Session: `{st.session_state.guest_thread_id}`")
        st.caption("Log in for full memory capabilities.")