from google import genai
client = genai.Client(api_key="Your_Gemini_API_KEY_fromGoogleAIstudio")

# here each prompt/question you ask is independent, no context information is retained/maintained from prev prompt in the next prompt
# e.g. if you tell the model your name is John in prompt 1, in prompt 2 it won't remember that your name is John
def talkToLLM(prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt
    )
    return response

def run():
    prompt = input("\nEnter your prompt: ")
    response = talkToLLM(prompt)
    print("-> ", response.text)
    run()

run()
