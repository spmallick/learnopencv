from actions import *
from tqdm import tqdm
from google import genai
import pyautogui, threading
from PIL import Image, ImageGrab
from collections import defaultdict
import io, os, re, time, json, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# Setup fail-safes
pyautogui.PAUSE = 2.5
pyautogui.FAILSAFE = True

# Store outputs from actions
action_outputs = defaultdict(dict)

predefined_paths = {
    "whatsapp": "whatsapp://",
    "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "edge": "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "sublime text": "C:/Program Files/Sublime Text 3/sublime_text.exe"
}


def read_text_from_image_gemini(client, query):
    # Capture screen
    screen_capture = ImageGrab.grab() 

    # Convert to bytes
    img_bytes_io = io.BytesIO()
    screen_capture.save(img_bytes_io, format="PNG") 
    img_bytes = img_bytes_io.getvalue()

    # Event to stop the progress bar
    stop_event = threading.Event()

    # Progress bar function
    def progress_task(stop_event):
        with tqdm(total=100, bar_format="⏳ Waiting for Gemini 2.5 Flash... {bar} {elapsed}") as pbar:
            while not stop_event.is_set():
                time.sleep(0.1)  # simulate waiting
                pbar.update(1)
                if pbar.n >= pbar.total:  # loop the bar
                    pbar.n = 0
                    pbar.last_print_n = 0
            pbar.close()

    # Start progress bar in a daemon thread
    thread = threading.Thread(target=progress_task, args=(stop_event,), daemon=True)
    thread.start()

    # Send request to Gemini
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": query},
                        {"inline_data": {"mime_type": "image/png", "data": img_bytes}},
                    ],
                }
            ],
        )
    finally:
        # Stop progress bar
        stop_event.set()
        thread.join()

    return response.text


def get_action_plan(client, prompt):
    system_prompt = f"""
    You are a gui-native agent planner. Convert the user's instruction into a JSON action plan.
    Use only the following action schema: 
      - read_text_from_image_gemini(client, query) 
      - locate_object_moondream(target_obj, model)
      - click(x=None, y=None, button="left")
      - double_click(x=None, y=None)
      - right_click(x=None, y=None)
      - move_mouse(x, y, duration=0.2)
      - press_key(key)
      - hotkey(*keys)
      - type_text(text)
      - clear_field()
      - launch_app(path)
      - sleep(seconds=None)
      - scroll(amount)

    Pre-defined paths (args) for launch_app action, windows specific:
      - "whatsapp" → will resolve to whatsapp://
      - "chrome"   → C:/Program Files/Google/Chrome/Application/chrome.exe
      - "edge"     → C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe
      - "sublime text" → "C:/Program Files/Sublime Text 3/sublime_text.exe"

    Apps other than predefined paths are to be searched onscreen or searched from start menu.
    Sleep time is in seconds. Always start by capturing screen, analysing what's on it, then move on to the next actions. 
    Add 2s delay after opening the application. For WhatsApp or Chrome or Telegram or other chat apps, once the app is open, start typing the user name directly. Press down arrow, enter, then type message.
    
    IMPORTANT: Any information retrieved from 'read_text_from_image_gemini' should be referenced in subsequent actions using the placeholder <OUTPUT_FROM_read_text_from_image_gemini>. Do NOT write fixed summaries. This placeholder will be replaced at runtime with the actual output.
    
    Each action must be a JSON object with keys:
      - "action": action name
      - "args": dictionary of arguments for that action
    
    Only output JSON array of actions. Do not include explanations or extra text.
    
    Instruction:
    {prompt}
    """
    print('Generating Action Plan. Please wait ...')

    # Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": system_prompt}]}],
    )
    print('✅ Done.')

    # Access text properly
    try:
        raw_text = response.candidates[0].content.parts[0].text.strip()
        raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
        action_plan = json.loads(raw_text)

        # Validate schema
        for i, step in enumerate(action_plan):
            if "action" not in step or "args" not in step:
                print(f"⚠️ Step {i} missing required keys:", step)

        return action_plan

    except Exception as e:
        print("❌ Error generating action plan:", e)
        print("Response text:", raw_text[:500])
        return []


def substitute_vars(arg_value):
    
    if isinstance(arg_value, str):
        # Replace {{var_name}} style placeholders
        matches = re.findall(r"{{(.*?)}}", arg_value)
        for m in matches:
            parts = m.split(".")
            var_name = parts[0]
            key = parts[1] if len(parts) > 1 else None
            value = action_outputs.get(var_name, {})
            if key:
                value = value.get(key, "")
            arg_value = arg_value.replace(f"{{{{{m}}}}}", str(value))
        
        # Replace special <OUTPUT_FROM_read_text_from_image_gemini> placeholder
        if "<OUTPUT_FROM_read_text_from_image_gemini>" in arg_value:
            gemini_output = action_outputs.get("read_text_from_image_gemini", {}).get("text", "")
            arg_value = arg_value.replace("<OUTPUT_FROM_read_text_from_image_gemini>", gemini_output)

        return arg_value

    elif isinstance(arg_value, list):
        return [substitute_vars(v) for v in arg_value]
    elif isinstance(arg_value, dict):
        return {k: substitute_vars(v) for k, v in arg_value.items()}
    else:
        return arg_value


def execute_actions(action_plan):
    """
    Execute a list of actions (from get_action_plan).
    """
    for step in action_plan:
        action_name = step.get("action")
        args = step.get("args", {})
        output_var = step.get("output")  # Optional output variable

        # Substitute variables in args (handles dynamic placeholders)
        args = {k: substitute_vars(v) for k, v in args.items()}

        try:
            result = None

            if action_name == "click_target":
                target = args["target"]
                coords = locate_object_moondream(target, moondream)
                if coords:
                    click(coords["x"], coords["y"])
                    result = coords
                else:
                    print(f"❌ Could not locate {target}")

            elif action_name == "read_text_from_image_gemini":
                query = args.get("query", "")
                result_text = read_text_from_image_gemini(client, query)
                result = {"text": result_text}

            elif action_name == "locate_object_moondream":
                desc = args["target_obj"]
                coords = locate_object_moondream(desc, moondream)
                result = coords if coords else {}

            elif action_name == "launch_app":
                app_path = args.get("path")
                launch_app(app_path)
                result = {"status": "launched"}

            elif action_name == "hotkey":
                keys = args.get("keys", [])
                if isinstance(keys, list):
                    pyautogui.hotkey(*keys)
                    result = {"pressed": keys}
                else:
                    print(f"❌ Invalid keys argument for hotkey: {keys}")

            else:
                # Map to local functions if available
                func = globals().get(action_name)
                if func:
                    result = func(**args) if args else func()
                else:
                    print(f"⚠️ Unknown action: {action_name}")

            # Store result in outputs
            if output_var:
                action_outputs[output_var] = result
            else:
                # Auto-store output from Gemini read if special action
                if action_name == "read_text_from_image_gemini":
                    action_outputs["read_text_from_image_gemini"] = result

            print(f"✅ Executed {action_name}, output: {result}")

        except Exception as e:
            print(f"❌ Error executing {action_name}: {e}")


def locate_object_moondream(target_obj, model):
    """
    Locate an object with Moondream and return the first detected point.
    """
    # Capture current screen
    screen_capture = ImageGrab.grab()  # PIL image
    height, width = screen_capture.height, screen_capture.width
    screen_capture.save("screen.png")

    # Ask Moondream to locate the target object
    response = model.point(screen_capture, target_obj)
    print("Model response:", response)

    try:
        points = response.get("points", [])
        if not points:
            raise ValueError("No points returned")

        # Pick the first point
        point = points[0]

        abs_x = int(point['x'] * width)
        abs_y = int(point['y'] * height)

        return {"x": abs_x, "y": abs_y}

    except Exception as e:
        print("Could not parse response. Got:", response, "Error:", e)
        return None


if __name__ == "__main__":
    time.sleep(0.5)

    parser = argparse.ArgumentParser(description="Agentic Action Plan Executor")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task prompt in natural language, e.g., 'Summarize screen content and send report via WhatsApp'"
    )
    args = parser.parse_args()
    task_to_do = args.task

    client = genai.Client()

    # # Load model
    # model_name = "moondream/moondream3-preview"
    # moondream = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     dtype=torch.bfloat16,
    #     device_map={"": "cuda"}
    # )
    # moondream.compile()

    # Get action plan from Gemini
    print("📝 Generating action plan ...")
    action_pln = get_action_plan(client, task_to_do)
    print("📋 Action plan:", action_pln)

    # 5. Execute actions
    execute_actions(action_pln)