import os
import re
from dotenv import load_dotenv
from prompt import prompt_template,FALLBACK_MESSAGE
import vector_store

load_dotenv()
# if even the best chunk is less similar than this, we refuse without asking the LLM
MIN_SCORE=float(os.getenv("MIN_SCORE","0.15"))

def get_llm():
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found in the .env file.")
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=api_key,model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),temperature=0)

def build_context(documents):
    # Put the retrieved chunks, with source and page, into one text block
    parts=[]
    for index,document in enumerate(documents,start=1):
        source=document.metadata["source"]
        page=document.metadata["page"]
        parts.append(f"Context {index}\nSource: {source}\nPage: {page}\nText:\n{document.page_content}")
    return "\n\n".join(parts)

def answer_question(question,index_folder=vector_store.INDEX_DIR):
    # 1. retrieve the most relevant chunks of the uploaded documents
    results=vector_store.search(question,index_folder=index_folder)
    sources=[{"source":d.metadata["source"],"page":d.metadata["page"],"score":round(float(s),3),"text":d.page_content} for d,s in results]

    # 2. nothing related found: refuse without calling the LLM
    if not results or max(s["score"] for s in sources)<MIN_SCORE:
        return {"answer":FALLBACK_MESSAGE,"refused":True,"sources":sources}

    # 3. ask the LLM, using only the retrieved context
    prompt=prompt_template.format(context=build_context([d for d,_ in results]),question=question)
    answer=get_llm().invoke(prompt).content.strip()
    # remove markdown images, so a malicious document cannot make the page load remote content
    answer=re.sub(r"!\[[^\]]*\]\([^)]*\)","",answer)
    refused="could not find this information" in answer.lower()
    return {"answer":answer,"refused":refused,"sources":sources}
