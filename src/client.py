import requests
import asyncio
import websockets


def test_uploading():
    url = "http://localhost:8000/upload"
    file_path = "src/NDA.pdf"

    with open(file_path, "rb") as file:
        file_content = file.read()

    files = {"file": ("file.pdf", file_content, "application/pdf")}

    response = requests.post(url, files=files)

    print(response.status_code)
    print(response.json())  


async def test_chat():
    uri = "ws://localhost:8000/chat"

    async with websockets.connect(uri) as websocket:

        print("Welcome to live chat!\n")

        while True:
            message = input("Enter your message (type 'exit' to quit): ")
            if message.lower() == 'exit':
                break

            await websocket.send(message)
            print(f"Message sent: {message}")

            response = await websocket.recv()
            print(f"Received response: {response}\n")


if __name__ == "__main__":
    asyncio.run(test_chat())
    # test_uploading()