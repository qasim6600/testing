import pdfplumber
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from groq import Groq
import time
import os

import pickle

def cache_manual_data(manual_name, normal_text, figure_text, tables, model, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{manual_name.replace(' ', '_')}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached data for {manual_name}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Processing and caching data for {manual_name}...")
    combined_text = normal_text + "\n" + figure_text
    text_chunks = chunk_text(combined_text)
    table_chunks = chunk_tables(tables)
    text_embeddings = model.encode(text_chunks)
    table_embeddings = model.encode(table_chunks)

    data = {
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "text_embeddings": text_embeddings,
        "table_embeddings": table_embeddings
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data

# --- Cleaning Function ---
def clean_text(text):
    text = re.sub(r'Page\s*\d+\s*', '', text, flags=re.IGNORECASE)  # Remove 'Page X'
    text = re.sub(r"(?<=\w)([A-Z]{2,})", r"\n\1", text)  # Break up long caps blocks
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines to one
    text = re.sub(r'[ \t]+', ' ', text)  # Remove extra spaces/tabs within lines
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

##def chunk_text(text, chunk_size=800, chunk_overlap=100):
##    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
##    return splitter.split_text(text)


##def chunk_text(text, max_chunk_size=800):
##    # Split text into sentences using punctuation
##    sentences = re.split(r'(?<=[.!?])\s+', text)
##    chunks = []
##    current_chunk = ""
##
##    for sentence in sentences:
##        if len(current_chunk) + len(sentence) <= max_chunk_size:
##            current_chunk += sentence + " "
##        else:
##            chunks.append(current_chunk.strip())
##            current_chunk = sentence + " "
##
##    if current_chunk:
##        chunks.append(current_chunk.strip())
##
##    return chunks


def chunk_text(text):
    # Match all-uppercase or Title Case lines as potential headings
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

# --- Model Load ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Groq Client Init ---
client = Groq(api_key="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")

# --- Manual Preloading ---
manuals = {
   # "ElectroLux washing Machine": "elctrolux.pdf",
   # "Mitsubishi Industrial AC": "Mitsubishi Industrial AC.pdf",
    "whirl-pool Microwave": "whirl-pool Microwave.pdf",
}

manual_data = {}

for product_name, pdf_file in manuals.items():
    try:
        normal_text, figure_text, tables = extract_manual_content(pdf_file)
        manual_data[product_name] = cache_manual_data(product_name, normal_text, figure_text, tables, model)
        print(f"✅ Loaded {product_name} | Text chunks: {len(manual_data[product_name]['text_chunks'])} | Table chunks: {len(manual_data[product_name]['table_chunks'])}")
    except Exception as e:
        print(f"⚠️ Failed loading {product_name}: {e}")

# --- Search Function ---
def search_chunks(query, embeddings, chunks, top_k=2):
    query_embed = model.encode(query)
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Validation Function ---
##def validate_query(query):
##    try:
##        response = client.chat.completions.create(
##            model="llama3-70b-8192",
##            messages=[
##                {"role": "system", "content": "You are an input validation assistant. Determine if the following query is relevant to the manual topics, such as installation, maintenance, or troubleshooting. Respond ONLY with 'Valid Question' if the query is relevant to any of the manuals' or 'Invalid Input'."},
##                {"role": "user", "content": f"Input: {query}"}
##            ],
##            temperature=0,
##            max_tokens=10
##        )
##        return response.choices[0].message.content.strip()
##    except Exception as e:
##        print(f"Validation error: {e}")
##        return "Valid Question"


def validate_query(query, manual_name):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a strict assistant validating user questions for the product manual titled '{manual_name}'.\n"
                        "If the question is relevant to installation, operation, troubleshooting, or safety of the product, respond with exactly: Valid Question.\n"
                        "If it is NOT relevant, you MUST respond in this exact format:\n"
                        "Invalid Input: Try questions like:\n"
                        "- [example 1]\n"
                        "- [example 2]\n"
                        "- [example 3]"
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0.4,
            max_tokens=120
        )

        result = response.choices[0].message.content.strip()

        if result.startswith("Valid Question"):
            return "Valid Question"
        elif result.startswith("Invalid Input:"):
            return result
        else:
            return "Invalid Input: Unexpected format."

    except Exception as e:
        print(f"Validation error: {e}")
        return "Valid Question"


# --- Context Formatter ---
def format_context_chunks(chunks):
    return "\n".join([f"- {chunk}" for chunk in chunks])

# --- Groq QA ---
def extract_answer(query, text_context, table_context, product_name):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": f"""
You are a professional technical assistant for {product_name}.
You MUST answer using ONLY the provided CONTEXT.
- Do NOT mention or reference the context types (like 'table content' or 'text').
- Read and analyze the CONTEXT deeply.
- Combine details from both TEXT and TABLE context if needed.
- Provide a complete and detailed answer that covers as much relevant information as possible from the CONTEXT.
- If answer is missing, say: \"Sorry, I could not find this information in the manual.\"
"""},
                {"role": "user", "content": f"QUERY: {query}\n\nTEXT CONTEXT:\n{format_context_chunks(text_context)}\n\nTABLE CONTEXT:\n{format_context_chunks(table_context)}"}
            ],
            temperature=0,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return text_context[:200] + "..."

# ... [imports and all previous logic stay the same above this]

with gr.Blocks(title="Product Manual Assistant", elem_id="main-container", css=""" 
    html, body {
        
        margin: 0;
        padding: 0;
        overflow: hidden;
        background-image: url('https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }
    #main-container {
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    .gr-row {
        flex: 1;
        overflow: hidden;
    }
    #header-block { background-color: rgba(245, 245, 245, 0.9); padding: 10px; border-radius: 8px; }
    #left-column  { background-color: rgba(227, 242, 253, 0.9); padding: 10px; border-radius: 8px; }
    #right-column { background-color: rgba(252, 228, 236, 0.9); padding: 10px; border-radius: 8px; }
    #left-column, #right-column {
        height: 100%;
        overflow-y: auto;
    }
    #user-input-box textarea {
        font-size: 16px;
    }
""") as demo:

    gr.Markdown("# Product Assistant\nSelect YOUR PRODUCT and ask questions ")

    with gr.Row(elem_id="header-block"):
        with gr.Column(scale=4):
            manual_selector = gr.Dropdown(
                choices=["Select your product"] + list(manual_data.keys()),
                label="Select Product Manual",
                value="Select your product"
            )
            chatbot = gr.Chatbot(value=[], label="", type="messages", bubble_full_width=False, height=350, elem_id="chat-window")
##            scroll_anchor = gr.HTML("<div id='scroll-anchor'></div>")
            user_input = gr.Textbox(placeholder="Ask a question about your product...", show_label=False, elem_id="user-input-box")
            send_btn = gr.Button("Ask", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Product Info")
            product_info = gr.Markdown("Select a product to see info.")
            show_extracted_checkbox = gr.Checkbox(label="Show Extracted Manual Content", value=False)
            extracted_content_box = gr.Markdown("")

    def update_product_info(product_name):
        product_details = {
            "ElectroLux washing Machine": "Product Info:\n- Product Name = Automatic Washing Machine\n- Type = Top Load\n- Capacity = 9Kg\n- Color = Grey and White",
            "Mitsubishi Industrial AC": "Product Info:\n- Product Name = AIR-CONDITIONER\n- Type and Model = Industrial AC SRK25ZMP-S, SRK35ZMP-S, SRK45ZMP-S",
            "whirl-pool Microwave": "Product Info:\n- Product Name = MICROWAVE OVEN\n- Type = COUNTERTOP MICROWAVE\n- Warranty = ONE YEAR LIMITED WARRANTY", }
        return product_details.get(product_name, "Product Info: Not available.")

    def respond(message, history, selected_product, show_extracted):
        if selected_product == "Select your product" or selected_product not in manual_data:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⚠ Please select a valid product from the dropdown."})
            yield history, "", ""
            return

        validation = validate_query(message,selected_product)
        if validation != "Valid Question":
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": validation})  # show suggestions!
            yield history, "", ""
            return
##        if validation != "Valid Question":
##            history.append({"role": "user", "content": message})
##            history.append({"role": "assistant", "content": "⚠ Please ask a proper question related to the product manual."})
##            yield history, "", ""
##            return

        data = manual_data[selected_product]
        top_text_chunks = search_chunks(message, data["text_embeddings"], data["text_chunks"], top_k=2)
        top_table_chunks = search_chunks(message, data["table_embeddings"], data["table_chunks"], top_k=2)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": " Thinking..."})
        yield history, "", ""
        time.sleep(0.05)

        answer = extract_answer(message, top_text_chunks, top_table_chunks, selected_product)
        history[-1]["content"] = answer

        extracted_text = "\n\n".join(top_text_chunks + top_table_chunks) if show_extracted else ""
        yield history, "", extracted_text

    def clear_chat():
        return [], "", ""
    def toggle_button_state(product):
        return gr.update(interactive=product != "Select your product")

    manual_selector.change(toggle_button_state, manual_selector, send_btn)

    manual_selector.change(fn=lambda product: ([], "", "", update_product_info(product)), 
                           inputs=manual_selector, 
                           outputs=[chatbot, user_input, extracted_content_box, product_info])

    send_btn.click(respond, inputs=[user_input, chatbot, manual_selector, show_extracted_checkbox], outputs=[chatbot, user_input, extracted_content_box], show_progress=True)
    user_input.submit(respond, inputs=[user_input, chatbot, manual_selector, show_extracted_checkbox], outputs=[chatbot, user_input, extracted_content_box], show_progress=True)

    gr.HTML("""
    <script>
        // Observe mutations in the chat container to auto-scroll
        const observer = new MutationObserver(() => {
            const chat = document.querySelector("#chat-window .wrap");
            if (chat) {
                chat.scrollTop = chat.scrollHeight;
            }
        });

        const chatContainer = document.querySelector("#chat-window .wrap");
        if (chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }

        // Ensure chat is observed again after updates
        const interval = setInterval(() => {
            const chat = document.querySelector("#chat-window .wrap");
            if (chat && !chat.hasAttribute("data-observed")) {
                observer.observe(chat, { childList: true, subtree: true });
                chat.setAttribute("data-observed", "true");
            }
        }, 1000);
    </script>
    """)


    

    demo.launch(server_name="0.0.0.0", server_port=80)

