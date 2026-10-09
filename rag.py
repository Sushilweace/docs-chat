import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
llm = Anthropic()  # reads ANTHROPIC_API_KEY from env


def chunk(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def ingest(pdf_path):
    name = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i, c in enumerate(chunk(text)):
            collection.add(
                ids=[f"{name}-p{page_num}-c{i}"],
                documents=[c],
                embeddings=[embedder.encode(c).tolist()],
                metadatas=[{"source": name, "page": page_num}],
            )


def ask(question, k=4):
    q_emb = embedder.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas)
    )

    msg = llm.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=800,
        system=(
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources like [file.pdf p.3]."
        ),
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return msg.content[0].text