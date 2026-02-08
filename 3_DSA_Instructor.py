from google import genai
from google.genai import types

client = genai.Client(api_key="Your_GEMINI_API_KEY_FromGoogleAIStudio")

# here each prompt/question you ask, the context information is retained/maintained from prev prompt in the next prompt
# e.g. if you tell the model your name is John in prompt 1, in prompt 2 it would remember that your name is John
instructions = """You are a DSA instructior, which helps students to learn Data Structures and Algorithms (DSA) concepts with easy to understand examples.
If someone asks you anything not related to DSA, you politely refuse to answer them and remind them that you are a DSA instructor and only answer DSA related questions."""

chat = client.chats.create(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction=instructions
    )
)

def talkToLLM(prompt):
    response = chat.send_message(prompt)
    return response

def run():
    prompt = input("\nEnter your prompt: ")
    response = talkToLLM(prompt)
    print("-> ", response.text)
    run()

run()
