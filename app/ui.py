import gradio as gr
from app.rag_backend import ingest_file, query_rag

def build_ui():
    with gr.Blocks(title="Gemini RAG Demo") as demo:
        gr.Markdown("## 📚 Gemini RAG System\nUpload docs and ask questions.")

        with gr.Row():
            file_upload = gr.File(label="Upload documents", file_types=[".pdf",".docx",".txt"])
            upload_btn = gr.Button("Ingest")
            upload_out = gr.Textbox(label="Ingestion status")

        upload_btn.click(ingest_file, inputs=file_upload, outputs=upload_out)

        with gr.Row():
            query_in = gr.Textbox(label="Ask a question", placeholder="Type your question...")
            query_btn = gr.Button("Search & Answer")
            query_out = gr.Textbox(label="Answer")

        query_btn.click(query_rag, inputs=query_in, outputs=query_out)

    return demo
