from openai import OpenAI
import os

os.environ['no_proxy'] = "10.140.0.231:18901"

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.140.0.231:18901/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="judge",
                                      prompt="San Francisco is a")
print("Completion result:", completion)