from fastapi import FastAPI
import uvicorn
import os
import pdfplumber
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from groq import Groq
import time

# --- Cleaning Function ---
def clean_text(text):
    text = re.sub(r'Page\s*\d+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\w)([A-Z]{2,})", r"\n\1", text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# --- Extraction Functions ---
def is_inside(small_box, big_box):
    sx0, sy0, sx1, sy1 = small_box
    bx0, by0, bx1, by1 = big_box
    return (sx0 >= bx0 and sy0 >= by0 and sx1 <= bx1 and sy1 <= by1)

def extract_manual_content(pdf_path):
    normal_texts = []
    figure_texts = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images = page.images
            words = page.extract_words()

            for word in words:
                word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
                inside_image = any(is_inside(word_bbox, (float(img['x0']), float(img['top']), float(img['x1']), float(img['bottom']))) for img in images)
                if inside_image:
                    figure_texts.append(word['text'])
                else:
                    normal_texts.append(word['text'])

            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    tables.append(table)

    normal_text = clean_text(" ".join(normal_texts))
    figure_text = clean_text(" ".join(figure_texts))

    return normal_text, figure_text, tables

def chunk_text(text):
    heading_pattern = re.compile(r'^(?:[A-Z][A-Z\s]{3,}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group().strip()
        body = text[start:end].strip()
        if body:
            chunks.append(f"{heading}\n{body}")

    return chunks

def chunk_tables(tables):
    table_chunks = []
    for table in tables:
        table_text = "\n".join([" | ".join([cell if cell else "" for cell in row]) for row in table])
        table_chunks.append(table_text)
    return table_chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key="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")

manual_data = {}

def search_chunks(query, embeddings, chunks, top_k=2):
    query_embed = model.encode(query)
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def validate_query(query):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a validation assistant. Respond only with 'Valid Question' or 'Invalid Input'."},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Valid Question"

def format_context_chunks(chunks):
    return "\n".join([f"- {chunk}" for chunk in chunks])

ef extract_answer(query, text_context, table_context):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Answer the question using the provided context only."},
                {"role": "user", "content": f"QUERY: {query}\n\nTEXT CONTEXT:\n{format_context_chunks(text_context)}\n\nTABLE CONTEXT:\n{format_context_chunks(table_context)}"}
            ],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq API error:", e)  # <--- SHOW actual error
        return "Sorry, something went wrong."


with gr.Blocks() as demo:
    with gr.Column():
        upload = gr.File(label="Upload Manual (PDF)", file_types=[".pdf"])
        chatbot = gr.Chatbot(type="messages")
        query = gr.Textbox(placeholder="Ask your question")
        ask_btn = gr.Button("Ask")

    def handle_manual(file):
        if file is None:
            return gr.update(visible=False), None
        normal_text, figure_text, tables = extract_manual_content(file.name)
        combined_text = normal_text + "\n" + figure_text
        text_chunks = chunk_text(combined_text)
        table_chunks = chunk_tables(tables)
        text_embeddings = model.encode(text_chunks)
        table_embeddings = model.encode(table_chunks)
        manual_data['current'] = {
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "text_embeddings": text_embeddings,
            "table_embeddings": table_embeddings
        }
        return gr.update(visible=True), ""

    
    def handle_query(user_msg, history):
        if 'current' not in manual_data:
            return history + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": "Please upload a manual first."}], ""

        validation = validate_query(user_msg)
        if validation != "Valid Question":
            return history + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": validation}], ""

        data = manual_data['current']
        top_text_chunks = search_chunks(user_msg, data["text_embeddings"], data["text_chunks"])
        top_table_chunks = search_chunks(user_msg, data["table_embeddings"], data["table_chunks"])
        answer = extract_answer(user_msg, top_text_chunks, top_table_chunks)
        return history + [{"role": "user", "content": user_msg},
                          {"role": "assistant", "content": answer}], ""

    upload.change(handle_manual, upload, [ask_btn, chatbot])
    ask_btn.click(handle_query, [query, chatbot], [chatbot, query])
    query.submit(handle_query, [query, chatbot], [chatbot, query])

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/gradio")
@app.get("/")
def read_root():
    return {"status": "OK"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
