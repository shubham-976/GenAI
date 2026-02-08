from pathlib import Path
import json
from google import genai
from google.genai import types

instructions = """
You are an AI website generator.

Your task:
- Generate frontend websites using HTML, CSS, and JavaScript.

IMPORTANT OUTPUT RULES:
- Return ONLY valid JSON
- Do NOT include markdown
- Do NOT include explanations
- Do NOT include shell commands
- Do NOT escape content unnecessarily

OUTPUT SCHEMA:
{
  "project_name": string,
  "files": [
    {
      "path": string,
      "content": string
    }
  ]
}

Each file must contain complete, valid code.
"""

client = genai.Client(api_key="YOUR_GEMINI_API_KEY_HERE_FromGoogleStudio")
chat = client.chats.create(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction=instructions
    )
)

def materialize_project(spec: dict, base_dir="."):
    root = Path(base_dir) / spec["project_name"]
    root.mkdir(parents=True, exist_ok=True)

    for file in spec["files"]:
        path = root / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file["content"], encoding="utf-8")

    return root

def talkToLLM(prompt: str):
    response = chat.send_message(prompt)
    # print("[DEBUG] Full response:", response)
    raw = response.text.strip()
    # print("[DEBUG] Raw response text:", raw)

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("Model did not return valid JSON") from e

    project_path = materialize_project(spec)
    return project_path


def run():
    while True:
        prompt = input("\nEnter your prompt: ")
        project_path = talkToLLM(prompt)
        print(f"\n✅ Website created at: {project_path.resolve()}")

run()
