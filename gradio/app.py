import json

import requests

import gradio as gr

# FastAPI backend URL
FASTAPI_URL = "http://web:8000"


def chat_with_llm(message, history):
    """Send message to FastAPI backend and return response"""
    try:
        # Make request to FastAPI backend
        response = requests.get(
            f"{FASTAPI_URL}/ask", params={"prompt": message}, timeout=180
        )

        if response.status_code == 200:
            # Parse the JSON response
            try:
                result = response.json()
                if "error" in result:
                    return f"Error: {result['error']}"

                # Extract the response text from the Ollama response
                if "response" in result:
                    return result["response"]
                else:
                    # If it's the raw Ollama response format
                    return result.get("response", str(result))

            except json.JSONDecodeError:
                # If response is plain text
                return response.text

        else:
            return f"API Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}"


def check_backend_health():
    """Check if the FastAPI backend is available"""
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


# Create Gradio interface
with gr.Blocks(title="Gemma3n Chat Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Gemma3n Chat Interface")
    gr.Markdown("Chat with the Gemma3n model running on Ollama!")

    # Add a status indicator
    with gr.Row():
        status_text = gr.Textbox(
            value="Checking backend status...",
            label="Backend Status",
            interactive=False,
            container=True,
        )

    # Main chat interface
    chatbot = gr.Chatbot(height=500, show_label=False, container=True, type="messages")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            label="Message",
            scale=4,
            container=False,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")
        clear_btn = gr.Button("Clear", scale=1, variant="secondary")

    # Example prompts
    with gr.Row():
        gr.Examples(
            examples=[
                "Hello! How are you?",
                "Explain quantum computing in simple terms",
                "Write a short poem about artificial intelligence",
                "What are the benefits of using Docker?",
                "Tell me a fun fact about space",
            ],
            inputs=msg,
            label="Example prompts",
        )

    def update_status():
        if check_backend_health():
            return "✅ Backend is running and healthy"
        else:
            return "❌ Backend is not responding"

    def respond(message, chat_history):
        # Add user message to history
        chat_history.append({"role": "user", "content": message})

        # Get response from LLM
        bot_response = chat_with_llm(message, chat_history)

        # Update the last message with bot response
        chat_history.append({"role": "assistant", "content": bot_response})

        return "", chat_history

    def clear_chat():
        return [], ""

    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

    # Update status on load
    demo.load(update_status, outputs=status_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
