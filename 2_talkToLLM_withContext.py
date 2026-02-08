from google import genai
client = genai.Client(api_key="Your_Gemini_API_KEY_from_GoogleAIStudio")

# here each prompt/question you ask, the context information is retained/maintained from prev prompt in the next prompt
# e.g. if you tell the model your name is John in prompt 1, in prompt 2 it would remember that your name is John
chat = client.chats.create(model="gemini-3-flash-preview")

def talkToLLM(prompt):
    response = chat.send_message(prompt)
    return response

def run():
    prompt = input("\nEnter your prompt: ")
    response = talkToLLM(prompt)
    print("-> ", response.text)
    run()

run()
