import gradio as gr
import subprocess
import time 

# --- Ollama Helper Functions ---

def check_ollama_running():
    """Checks if the Ollama service is accessible."""
    try:
        subprocess.run(["ollama", "ps"], check=True, capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_ollama_models():
    """Gets a list of locally available Ollama models."""
    if not check_ollama_running():
        return []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        models = []
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

# --- Core Logic ---

# Typing speed simulation
CHAR_DELAY = 0.02  # Adjust for desired speed (0.01 is fast, 0.05 is slower)

def reasoning_ollama_stream(model_name, prompt, mode):
    """
    Streams response from an Ollama model with simulated typing speed.
    """
    if not model_name:
        yield "Error: No model selected. Please choose a model."
        return
    if not prompt.strip():
        yield "Error: Prompt cannot be empty."
        return

    if not check_ollama_running():
        yield "Error: Ollama service does not seem to be running or accessible. Please start Ollama."
        return

    available_models = get_ollama_models()
    if model_name not in available_models:
        yield f"Error: Model '{model_name}' not found locally. Please pull it using 'ollama pull {model_name}'."
        return

    prompt_with_mode = f"{prompt.strip()} /{mode}"
    command = ["ollama", "run", model_name]

    displayed_response = ""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        process.stdin.write(prompt_with_mode + "\n")
        process.stdin.close()

        # Stream stdout, simulating typing
        for line_chunk in iter(process.stdout.readline, ''):
            if not line_chunk: # Should be caught by iter's sentinel, but defensive
                break
            for char in line_chunk:
                displayed_response += char
                yield displayed_response # Yield the progressively built response
                time.sleep(CHAR_DELAY)  # Pause briefly after each character

        process.stdout.close()
        return_code = process.wait()

        # Yield any remaining part of displayed_response if an error occurs after some output
        if return_code != 0:
            error_output = process.stderr.read()
            error_message = f"\n\n--- Ollama Error (code {return_code}) ---\n{error_output.strip()}"
            if displayed_response and not displayed_response.endswith(error_message): # Avoid double printing if already part of stream
                 displayed_response += error_message
                 yield displayed_response
            elif not displayed_response:
                 yield error_message.strip() # Only error
            return # Important to return after handling error

        if not displayed_response.strip() and return_code == 0:
             yield "Model returned an empty response."
        elif displayed_response: # Ensure final complete response is yielded if no error
            yield displayed_response


    except FileNotFoundError:
        yield "Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH."
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"

# --- Gradio UI ---

AVAILABLE_MODELS = get_ollama_models()
QWEN_MODELS = [m for m in AVAILABLE_MODELS if "qwen" in m.lower()] # Case-insensitive Qwen check
INITIAL_MODEL = None

# Prioritize qwen3:8b if available
if "qwen3:8b" in AVAILABLE_MODELS:
    INITIAL_MODEL = "qwen3:8b"
elif QWEN_MODELS: # Then any other Qwen model
    INITIAL_MODEL = QWEN_MODELS[0]
elif AVAILABLE_MODELS: # Then any available model
    INITIAL_MODEL = AVAILABLE_MODELS[0]

with gr.Blocks(title="Qwen3 x Ollama", theme=gr.themes.Soft()) as demo:
    
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Qwen3 Reasoning with Ollama
        </h1>
    """)
    
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://opencv.org/university/' target='_blank'>OpenCV Courses</a> | <a href='https://github.com/OpenCV-University' target='_blank'>Github</a>
        </h3>
        """
    )
    
    gr.Markdown(
        """
        - Interact with a Qwen3 model hosted on Ollama.
        - Switch between `/think` and `/no_think` modes to explore the thinking process.
        - The response will stream with a simulated typing effect.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=AVAILABLE_MODELS if AVAILABLE_MODELS else ["No models found - check Ollama"],
                value=INITIAL_MODEL,
                interactive=True
            )
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="e.g., Explain quantum entanglement in simple terms.",
                lines=5,
                elem_id="prompt-input" # For clear button targeting
            )
            mode_radio = gr.Radio(
                ["think", "no_think"],
                label="Reasoning Mode",
                value="think",
                info="`/think` encourages step-by-step reasoning. `/no_think` aims for a direct answer."
            )
            with gr.Row():
                submit_button = gr.Button("Generate Response", variant="primary")
                # ClearButton can now target specific components by elem_id or reference
                clear_button = gr.ClearButton()


        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="Status", interactive=False, lines=1,
                placeholder="Awaiting submission...",
                elem_id="status-output" # For clear button targeting
            )
            response_output = gr.Textbox(
                label="Model Response",
                lines=20, # Increased lines for better viewing
                interactive=False,
                show_copy_button=True,
                elem_id="response-output" # For clear button targeting
            )

    # Linking components
    def handle_submit_wrapper(model, prompt, mode):
        # Initial status update
        yield {status_output: "Processing... Preparing to stream response.", response_output: ""}

        # Stream the actual response
        # The generator from reasoning_ollama_stream will update response_output
        # The last yielded value from reasoning_ollama_stream will be the final full_response or an error.
        final_chunk = ""
        for chunk in reasoning_ollama_stream(model, prompt, mode):
            final_chunk = chunk # Keep track of the latest state
            yield {status_output: "Streaming response...", response_output: chunk}

        # Final status update based on the content of the last chunk
        if "Error:" in final_chunk or "--- Ollama Error ---" in final_chunk:
             yield {status_output: "Completed with issues.", response_output: final_chunk}
        elif "Model returned an empty response." in final_chunk:
            yield {status_output: "Model returned an empty response.", response_output: final_chunk}
        elif not final_chunk.strip() and ("Error:" not in final_chunk and "--- Ollama Error ---" not in final_chunk):
            # This case might occur if stream ends abruptly or only yields empty strings
            yield {status_output: "Completed, but no substantive output received.", response_output: final_chunk}
        else:
            yield {status_output: "Response generated successfully!", response_output: final_chunk}


    submit_button.click(
        fn=handle_submit_wrapper,
        inputs=[model_selector, prompt_input, mode_radio],
        outputs=[status_output, response_output]
    )
    # Configure ClearButton to clear the relevant fields
    clear_button.add([prompt_input, response_output, status_output])


    gr.Examples(
        examples=[
            # Provide the initial model if available, otherwise let user pick
            [INITIAL_MODEL if INITIAL_MODEL else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen3:8b"), "What are the main pros and cons of using nuclear energy?", "think"],
            [INITIAL_MODEL if INITIAL_MODEL else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen3:4b"), "Write a short poem about a rainy day.", "no_think"],
            [INITIAL_MODEL if INITIAL_MODEL else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen3:8b"), "Plan a 3-day trip to Paris, focusing on historical sites.", "think"],
        ],
        inputs=[model_selector, prompt_input, mode_radio],
        outputs=[status_output, response_output],
        fn=handle_submit_wrapper,
        cache_examples=False
    )
    
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Developed with ❤️ by OpenCV
        </h3>
        """
    )

if __name__ == "__main__":
    print("Attempting to fetch Ollama models...")
    if not AVAILABLE_MODELS:
        if check_ollama_running():
            print("Warning: Ollama is running, but no models were found or could be listed.")
            print("The UI might not function correctly. Try pulling a model, e.g., 'ollama pull qwen2:7b'")
        else:
            print("CRITICAL ERROR: Ollama service does not seem to be running or accessible.")
            print("Please ensure Ollama is installed and running, then restart this application.")
            print("The application will attempt to launch, but model selection will be empty.")

    demo.queue().launch()