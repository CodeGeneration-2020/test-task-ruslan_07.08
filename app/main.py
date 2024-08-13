import os
import tempfile
import uvicorn
import openai
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from src.embedding_service import (
    initialize_pinecone,
    load_document,
    split_document,
    get_embedding,
    embed_and_store_document,
)

app = FastAPI()
index = initialize_pinecone()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = openai_api_key

user_sessions = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await manager.connect(websocket)
    user_sessions[websocket] = {"context": []}

    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("/reset"):
                user_sessions[websocket]["context"] = []
                await websocket.send_text("Context reset.")
                continue

            use_context = not data.startswith("/nocontext ")
            query = data[len("/nocontext "):] if not use_context else data

            response = await generate_response(query, user_sessions[websocket], use_context)
            await manager.send_personal_message(response, websocket)

            if use_context:
                user_sessions[websocket]["context"].extend([query, response])

            print(f"\n\n[CURRENT CHAT CONTEXT] {user_sessions[websocket]}\n\n")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        del user_sessions[websocket]


async def generate_response(query: str, session: dict, use_context: bool) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
    )

    matches = vectorstore.similarity_search(query)
    retrieved_content = [match.page_content for match in matches]
    context = "\n".join(retrieved_content)

    if use_context:
        chat_context = " ".join(session["context"])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\nUser's query: {query}\nChat context: {chat_context}"}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\nUser's query: {query}"}
        ]

    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    response_content = openai_response.choices[0].message['content']

    return response_content


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_type = file.content_type
        if file_type not in [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]:
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

        return {
            "filename": file.filename,
            "message": "File uploaded and processed successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Launched with 'poetry run start' at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
