from google import genai
from google.genai import types

# -----------------------------
# 1. Defin your Normal Python functions for the agent to use
# -----------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def sum_two_numbers(a: int, b: int) -> int:
    return a + b


def get_weather(city: str) -> str:
    # Dummy weather (replace with real API later)
    return f"The weather in {city} is sunny with 30°C."


# -----------------------------
# 2. Tool (function) declarations which tells the LLM about the functions descriptions and parameters
# -----------------------------

is_prime_declaration = {
        "name": "is_prime",
        "description": "Check whether a number is prime",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number to check"
                }
            },
            "required": ["n"]
        }
    }

sum_two_numbers_declaration = {
        "name": "sum_two_numbers",
        "description": "Find the sum of two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }
get_weather_declaration = {
        "name": "get_weather",
        "description": "Get weather information for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    }


# -----------------------------
# 3. Create LLM client + chat
# -----------------------------

client = genai.Client(api_key="YOUR_GEMINI_API_KEY_HERE")

chat = client.chats.create(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[is_prime_declaration, sum_two_numbers_declaration, get_weather_declaration])]
    )
)


# -----------------------------
# 4. Core agent loop : AI agent to talk to LLM, check for tool calls, execute tools, and return results to LLM and final response to user
# -----------------------------

def talkToLLM(prompt: str) -> str:
    response = chat.send_message(prompt)
    counter = 1
    while True:
        print("counter :", counter)
        counter += 1
        parts = response.candidates[0].content.parts
        # print("[DEBUG] response : ", response)
        # print("[DEBUG] response.candidates[0] : ", response.candidates[0])
        # print("[DEBUG] response.candidates[0].content : ", response.candidates[0].content)
        # print("[DEBUG] response.candidates[0].content.parts : ", response.candidates[0].content.parts)
        tool_called = False

        for part in parts:
            # print("step-1 part : ", part)
            if part.function_call:
                # print("step-2 function call : ", part.function_call)
                tool_called = True
                fc = part.function_call
                name = fc.name
                args = fc.args

                # Execute the requested function
                if name == "is_prime":
                    result = is_prime(args["n"])

                elif name == "sum_two_numbers":
                    result = sum_two_numbers(args["a"], args["b"])

                elif name == "get_weather":
                    result = get_weather(args["city"])

                # print("Hi\n")
                # Send tool result back to the LLM
                response = chat.send_message(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"result": result}
                        )
                    )
                )


                break  # important: re-check model response

        # If no tool was requested, final answer is ready
        if not tool_called:
            return response.text


# -----------------------------
# 5. Interactive loop to talk to the agent

# -----------------------------

def run():
    while True:
        prompt = input("\nEnter your prompt: ")
        reply = talkToLLM(prompt)
        print("->", reply)


run()
