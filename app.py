import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL    = "llama3"
VECTOR_STORE    = "faiss_index"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


# ── PROMPT TEMPLATE ──────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not found in the context, say "I don't have enough information to answer that."
Do NOT make up answers.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def load_pdf(pdf_path: str):
    print(f"\n📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    print("\n🔍 Generating embeddings with HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE)
    print(f"   Vector store saved to '{VECTOR_STORE}/'")
    return vector_store


def load_vector_store():
    print("\n📦 Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def build_rag_chain(vector_store):
    print("\n🤖 Connecting to Ollama LLM (llama3)...")
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4} 
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def query_loop(chain):
    print("\n" + "="*60)
    print("Riverr-RAG: PDF Q&A System")
    print(" Type your question | 'quit' to exit")
    print("="*60)

    while True:
        question = input("\n❓ Your Question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue

        print("\n⏳ Thinking...")
        result = chain.invoke({"query": question})

        print(f"\n💡 Answer:\n{result['result']}")

        print("\n📎 Sources used:")
        for i, doc in enumerate(result["source_documents"], 1):
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"   [{i}] Page {page + 1}: {snippet}...")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app.py <path_to_pdf>")
        print("Example: python app.py documents/sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found — {pdf_path}")
        sys.exit(1)

    if os.path.exists(VECTOR_STORE):
        print(f"\n✅ Found existing vector store. Loading...")
        vector_store = load_vector_store()
    else:
        chunks = load_pdf(pdf_path)
        vector_store = build_vector_store(chunks)

    chain = build_rag_chain(vector_store)

    query_loop(chain)


if __name__ == "__main__":
    main()
