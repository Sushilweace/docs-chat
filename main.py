import shutil
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from rag import ingest, ask

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile):
    path = f"./{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ingest(path)
    return {"status": "ingested", "file": file.filename}


@app.post("/ask")
async def ask_question(body: Question):
    return {"answer": ask(body.question)}