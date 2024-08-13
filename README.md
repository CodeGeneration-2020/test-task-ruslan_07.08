# Live Chat

### Installation
1. Clone the repository:
```bash
git clone [project url]
```
2. Install dependencies:
```bash
poetry install
```
3. Set up environment variables:
Create a .env file at the root of your project with the following variables:
```bash
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index-name
OPENAI_API_KEY=your-openai-api-key
```
4. Run the application:
Start the server with:
```bash
poetry run python man.py
```
The server will run on http://localhost:8000.

### API Endpoints
##### WebSocket Endpoint
/chat: The WebSocket endpoint for the live chat. Users can send messages and receive responses in real-time.
Supported Commands
/reset: Resets the chat context. This clears all previous interactions and starts a new session.
/nocontext: Sends a query without using the stored context. The response is generated solely based on the current query.
##### REST API Endpoints
- POST /upload: Uploads and processes a document to store in the Pinecone vector database.
- Accepted File Types: application/pdf, application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain
Response:
```json
{
    "filename": "your_file.pdf",
    "message": "File uploaded and processed successfully"
}
```

### Usage
##### Chat Interaction
1. Connect to the WebSocket:
Use a WebSocket client (or a custom client) to connect to ws://localhost:8000/chat.
2. Send Messages:
- Send any text message to interact with the AI assistant.
- Use /reset to clear the session context.
- Use /nocontext your_message to bypass the stored context.
3. Receive Responses:
The AI assistant will respond to your queries in real-time, utilizing both the current context and the RAG mechanism.

### Uploading Documents
Upload a document:
Use the /upload endpoint to upload a document for processing.
```bash
curl -X POST "http://localhost:8000/upload" -F "file=@your_file.pdf"
```
Process the document:
The document is split into chunks, embedded using a transformer model, and stored in Pinecone for future retrieval during chat interactions.
