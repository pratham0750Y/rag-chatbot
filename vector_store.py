import shutil
from functools import lru_cache
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import document_loader

EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
# folder where the FAISS index is saved
INDEX_DIR=Path("vector_store/saved_index")
# normalized embeddings + inner product = cosine similarity (higher is better)
STRATEGY=DistanceStrategy.MAX_INNER_PRODUCT

@lru_cache(maxsize=1)
def get_embeddings():
    # imported here and cached, so the model is loaded only once
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,encode_kwargs={"normalize_embeddings":True})

def rebuild_index(pdf_folder,index_folder=INDEX_DIR):
    # Read all PDFs of the folder, split into chunks, embed them and save the FAISS index.
    # Rebuilding is simple and fast enough for a few documents.
    shutil.rmtree(index_folder,ignore_errors=True)
    documents=[]
    for pdf in sorted(Path(pdf_folder).glob("*.pdf")):
        documents+=document_loader.load_pdf(pdf,pdf.name)
    if not documents:
        return
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=120,separators=["\n\n","\n",". "," ",""])
    chunks=splitter.split_documents(documents)
    store=FAISS.from_documents(chunks,get_embeddings(),distance_strategy=STRATEGY)
    store.save_local(str(index_folder))

def search(question,k=4,index_folder=INDEX_DIR):
    # Return the k most similar chunks as (chunk, score) pairs
    if not Path(index_folder).exists():
        return []
    # the index files were created by this app, so loading them is safe
    store=FAISS.load_local(str(index_folder),get_embeddings(),allow_dangerous_deserialization=True,distance_strategy=STRATEGY)
    return store.similarity_search_with_score(question,k=k)
