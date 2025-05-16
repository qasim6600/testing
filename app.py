import gradio as gr

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
                choices=["Select your product", "ElectroLux washing Machine", "Mitsubishi Industrial AC", "whirl-pool Microwave"],
                label="Select Product Manual",
                value="Select your product"
            )
            chatbot = gr.Chatbot(value=[], label="", type="messages", bubble_full_width=False, height=350, elem_id="chat-window")
            user_input = gr.Textbox(placeholder="Ask a question about your product...", show_label=False, elem_id="user-input-box")
            send_btn = gr.Button("Ask", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Product Info")
            product_info = gr.Markdown("Select a product to see info.")
            show_extracted_checkbox = gr.Checkbox(label="Show Extracted Manual Content", value=False)
            extracted_content_box = gr.Markdown("")

    demo.launch(server_name="0.0.0.0", server_port=80)

