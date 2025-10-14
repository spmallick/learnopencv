import time
import asyncio
import gradio as gr
import numpy as np
from PIL import Image
from bleak import BleakClient

# BLE configuration

DEVICE_ADDR = "84:45:7d:35:39:74"  # Replace with your board's BLE MAC
IMG_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"     # image write characteristic
RESULT_UUID = "19b10002-e8f2-537e-4f6c-d104768a1214"  # result notify characteristic

TARGET_SIZE = (28, 28)
PREVIEW_SIZE = (128, 128)
CHUNK = 128  # BLE write chunk size in bytes

# Send image to BLE + wait for prediction
async def send_image_ble(image_path):
    # Load image and resize to match model input
    img = Image.open(image_path).convert("L").resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.uint8)

    # Convert to bytes for BLE transfer
    data_bytes = arr.tobytes()

    async with BleakClient(DEVICE_ADDR) as client:
        if not client.is_connected:
            raise Exception("❌ BLE connection failed")

        print("✅ Connected to BLE device")

        result_text = None

        # Callback for inference result
        def callback(sender, data):
            nonlocal result_text
            try:
                result_text = data.decode(errors="ignore").strip()
                print("📥 Received result:", result_text)
            except Exception as e:
                print("⚠️ Decode error:", e)

        await client.start_notify(RESULT_UUID, callback)

        # Send image in chunks (BLE-safe)
        print("📤 Sending image data...")
        for i in range(0, len(data_bytes), CHUNK):
            await client.write_gatt_char(IMG_UUID, data_bytes[i:i+CHUNK], response=False)
            await asyncio.sleep(0.03)

        # 4️⃣ Wait for MCU inference result
        print("⏳ Waiting for inference result...")
        for _ in range(100):
            if result_text:
                break
            await asyncio.sleep(0.05)

        await client.stop_notify(RESULT_UUID)

        if result_text is None:
            result_text = "⚠️ No response from MCU"

        return result_text, img.resize(PREVIEW_SIZE).convert("L")



def send_image_sync(image_path):
    """Synchronous wrapper for Gradio callback"""
    return asyncio.run(send_image_ble(image_path))


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 CNN Digit Classifier over BLE (Arduino Nano 33 BLE)")
    gr.Markdown(
        "Upload a **grayscale image (28×28)** — it’ll be quantized and sent via BLE. "
        "Your CNN model on Arduino performs inference and returns the predicted digit."
    )
    gr.Image("../../arduino-nano-33-BLE.jpg", show_label=False, elem_id="banner")

    with gr.Row():
        inp = gr.Image(type="filepath", label="Upload Image")
        out_text = gr.Textbox(label="Predicted Digit / Confidence")
        out_preview = gr.Image(label="Preprocessed 28×28 Preview")

    inp.change(fn=send_image_sync, inputs=inp, outputs=[out_text, out_preview])

if __name__ == "__main__":
    demo.launch()
