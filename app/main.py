from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import tempfile
import uvicorn
from src.embedding_service import initialize_pinecone, load_document, split_document, embed_and_store_document

app = FastAPI()
initialize_pinecone()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_type = file.content_type
        if file_type not in ["application/pdf",
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                             "text/plain"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        documents = load_document(temp_file_path, file_type)
        os.remove(temp_file_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load document")

        docs = split_document(documents)
        embed_and_store_document(docs, os.getenv("PINECONE_INDEX_NAME"))

        return {"filename": file.filename, "message": "File uploaded and processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Launched with 'poetry run start' at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
