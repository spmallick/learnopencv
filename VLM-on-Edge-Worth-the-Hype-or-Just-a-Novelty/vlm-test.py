import ollama
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run an Ollama vision-language model with image + query")
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b", help="Model name (default: qwen2.5vl:3b)")
    parser.add_argument("--image", type=str, default="./tasks/esp32-devkitC-v4-pinout.png", help="Path to input image")
    parser.add_argument("--query", type=str, default="Describe the contents of this image in 100 words.", help="Query string for the model")
    args = parser.parse_args()

    start_time = time.time()

    response = ollama.chat(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": args.query,
                "images": [args.image],
            }
        ]
    )

    end_time = time.time()

    print("Model Output:\n", response["message"]["content"])
    print("\nGeneration Time: {:.2f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
