# Offline tests: a fake embedding model and a fake LLM replace MiniLM and Groq.
# Run with:  pip install pytest   then   pytest
import re
import shutil
import sys
import zlib
from pathlib import Path
import pytest
from langchain_core.embeddings import Embeddings

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import document_loader
import evaluate
import rag_pipeline
import vector_store

DOCS=ROOT/"documents"

class FakeEmbeddings(Embeddings):
    # bag-of-words vectors: similar words give similar vectors, no download needed
    def embed_query(self,text):
        vector=[0.0]*1024
        for word in re.findall(r"[a-z0-9]+",text.lower()):
            vector[zlib.crc32(word.encode())%1024]+=1
        length=sum(v*v for v in vector)**0.5 or 1
        return [v/length for v in vector]
    def embed_documents(self,texts):
        return [self.embed_query(t) for t in texts]

class FakeMessage:
    def __init__(self,content):
        self.content=content

class FakeLLM:
    def invoke(self,prompt):
        return FakeMessage("Answer from the documents: ![x](http://evil.test/a.png) done")

@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(vector_store,"get_embeddings",lambda:FakeEmbeddings())
    monkeypatch.setattr(rag_pipeline,"get_llm",lambda:FakeLLM())

@pytest.fixture
def index(tmp_path):
    # an index built from the sample PDFs
    folder=tmp_path/"index"
    vector_store.rebuild_index(DOCS,folder)
    return folder

def test_check_pdf():
    good=(DOCS/"Policy.pdf").read_bytes()
    assert document_loader.check_pdf("Policy.pdf",good) is None
    assert document_loader.check_pdf("notes.txt",good)
    assert document_loader.check_pdf("fake.pdf",b"not a pdf")
    assert document_loader.check_pdf("empty.pdf",b"")
    assert document_loader.check_pdf("../evil.pdf",good)

def test_loader_keeps_source_and_page():
    pages=document_loader.load_pdf(DOCS/"Policy.pdf","Policy.pdf")
    assert [p.metadata["page"] for p in pages]==list(range(1,11))
    assert pages[0].metadata["source"]=="Policy.pdf"

def test_retrieval_finds_the_right_page(index):
    reply=rag_pipeline.answer_question("What is the leave policy?",index)
    assert (reply["sources"][0]["source"],reply["sources"][0]["page"])==("Policy.pdf",6)
    assert not reply["refused"]
    # markdown images from the model are removed
    assert "evil.test" not in reply["answer"]

def test_unrelated_question_is_refused(index):
    reply=rag_pipeline.answer_question("Volcano eruption Iceland lava",index)
    assert reply["refused"] and "could not find" in reply["answer"]

def test_no_index_means_refusal(tmp_path):
    reply=rag_pipeline.answer_question("What is the leave policy?",tmp_path/"missing")
    assert reply["refused"] and reply["sources"]==[]

def test_deleting_a_document_removes_it_from_the_index(tmp_path):
    pdfs=tmp_path/"pdfs"
    pdfs.mkdir()
    shutil.copy(DOCS/"Policy.pdf",pdfs)
    shutil.copy(DOCS/"Handbook.pdf",pdfs)
    index=tmp_path/"index"
    vector_store.rebuild_index(pdfs,index)
    (pdfs/"Policy.pdf").unlink()
    vector_store.rebuild_index(pdfs,index)
    reply=rag_pipeline.answer_question("What is the leave policy?",index)
    assert all(s["source"]=="Handbook.pdf" for s in reply["sources"])
    (pdfs/"Handbook.pdf").unlink()
    vector_store.rebuild_index(pdfs,index)
    assert not index.exists()

def test_evaluation_check():
    case={"expected_source":"Policy.pdf","expected_page":"6","should_refuse":"false","expected_keywords":"24"}
    good={"answer":"You get 24 days","refused":False,"sources":[{"source":"Policy.pdf","page":6}]}
    wrong_page={"answer":"You get 24 days","refused":False,"sources":[{"source":"Policy.pdf","page":5}]}
    assert evaluate.check(case,good)["passed"]
    assert not evaluate.check(case,wrong_page)["passed"]
    ceo={"expected_source":"","expected_page":"","should_refuse":"true","expected_keywords":""}
    assert evaluate.check(ceo,{"answer":"x","refused":True,"sources":[]})["passed"]
    assert not evaluate.check(ceo,{"answer":"Bob","refused":False,"sources":[]})["passed"]
