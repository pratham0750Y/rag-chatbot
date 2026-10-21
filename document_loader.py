import io
import re
from pypdf import PdfReader
from langchain_core.documents import Document

MAX_FILE_SIZE=20*1024*1024
# simple file names only (no folders), so a file cannot be saved outside the upload folder
SAFE_NAME=re.compile(r"^[\w .()-]{1,100}\.pdf$",re.IGNORECASE)

def load_pdf(file,filename):
    # Read every page and keep the file name and page number as metadata
    documents=[]
    reader=PdfReader(file)
    for page_number,page in enumerate(reader.pages,start=1):
        text=page.extract_text() or ""
        # skip empty pages safely
        if text.strip():
            documents.append(Document(page_content=text,metadata={"source":filename,"page":page_number}))
    return documents

def check_pdf(filename,data):
    # Validate an upload. Returns an error message, or None when the file is fine.
    if not SAFE_NAME.match(filename):
        return "Only .pdf files with a simple name are allowed"
    if not data:
        return "The file is empty"
    if len(data)>MAX_FILE_SIZE:
        return "The file is larger than 20 MB"
    # check the content, not only the extension
    if b"%PDF-" not in data[:1024]:
        return "Not a real PDF file"
    try:
        pages=load_pdf(io.BytesIO(data),filename)
    except Exception:
        return "The PDF could not be read"
    if not pages:
        return "No readable text found (scanned PDFs are not supported)"
    return None
