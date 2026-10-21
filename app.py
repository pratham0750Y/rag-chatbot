from pathlib import Path
import streamlit as st
import document_loader
import rag_pipeline
import vector_store

# uploaded PDFs are saved here, the FAISS index is saved in vector_store/saved_index
UPLOAD_DIR=Path("data/uploads")
st.set_page_config(page_title="RAG Chatbot",page_icon="📚")

# chat history must survive Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages=[]
if "upload_count" not in st.session_state:
    st.session_state.upload_count=0

def process_documents(uploaded):
    # Validate and save every uploaded PDF (same name = replace), then rebuild the index
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    saved=False
    for file in uploaded:
        data=file.getvalue()
        error=document_loader.check_pdf(file.name,data)
        if error:
            st.sidebar.error(file.name+": "+error)
        else:
            (UPLOAD_DIR/file.name).write_bytes(data)
            st.sidebar.success(file.name+" processed")
            saved=True
    if saved:
        with st.spinner("Reading and indexing..."):
            vector_store.rebuild_index(UPLOAD_DIR)
        # changing the key resets the uploader (clears the selected files)
        st.session_state.upload_count+=1

def sidebar():
    st.sidebar.header("Documents")
    uploaded=st.sidebar.file_uploader("Upload PDF files",type=["pdf"],accept_multiple_files=True,key="up_"+str(st.session_state.upload_count))
    if st.sidebar.button("Process documents"):
        if uploaded:
            process_documents(uploaded)
        else:
            st.sidebar.warning("Choose at least one PDF first.")

    # list of uploaded documents, each with a delete button
    for pdf in sorted(UPLOAD_DIR.glob("*.pdf")):
        left,right=st.sidebar.columns([4,1])
        left.caption("📄 "+pdf.name)
        if right.button("🗑",key="del_"+pdf.name):
            pdf.unlink()
            vector_store.rebuild_index(UPLOAD_DIR)
            st.rerun()

def show_answer(message):
    st.write(message["content"])
    if message["sources"]:
        with st.expander("Sources"):
            for number,source in enumerate(message["sources"],start=1):
                st.markdown(f"**{number}. {source['source']} - page {source['page']}** (similarity {source['score']})")
                # st.text shows the document text as plain text, never as markdown/HTML
                st.text(source["text"])

def chat_page():
    st.title("📚 RAG Chatbot")
    st.caption("Answers come only from your uploaded documents and can still be wrong. Please verify high-stakes information.")
    if st.button("Clear chat"):
        st.session_state.messages=[]
        st.rerun()

    # show the earlier messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"]=="user":
                st.write(message["content"])
            else:
                show_answer(message)

    question=st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the documents..."):
                    reply=rag_pipeline.answer_question(question)
            except ValueError as error:
                # for example: GROQ_API_KEY missing in .env
                st.error(str(error))
                return
            except Exception:
                st.error("The language model request failed. Please try again.")
                return
            message={"role":"assistant","content":reply["answer"],"sources":reply["sources"]}
            show_answer(message)
            st.session_state.messages.append(message)

sidebar()
chat_page()
