from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the OpenAI API key from environment variables

# Define the few-shot examples as part of the prompt
few_shot_prompt = """
Below are descriptions of different animals' daily habits. Provide a response in a similar style if asked about a new animal.

Example 1:
Animal: Dolphin
Response: Dolphins are known for their playful nature and often swim in groups called pods. They hunt for fish during the day, using echolocation to locate their prey, and spend several hours resting, but they never fully sleep. Instead, they rest one half of their brain at a time.

Example 2:
Animal: Owl
Response: Owls are nocturnal hunters. They sleep during the day, hidden in trees or barns, and become active at night. With their keen night vision, they hunt small mammals, using their silent flight to sneak up on prey.

Example 3:
Animal: Penguin
Response: Penguins are social birds, often found in large colonies. They spend much of their day hunting for fish in the water, diving deep and swimming fast. After feeding, they return to their nesting grounds to care for their young and interact with other penguins.

Animal: Fox
Response: 
"""

# Create a conversation using the ChatCompletion API with gpt-3.5-turbo
response = client.chat.completions.create(model="gpt-4-turbo",
messages=[
    {"role": "system", "content": "You are a helpful assistant that knows about animal behaviors."},
    {"role": "user", "content": few_shot_prompt}
],
max_tokens=100,
temperature=0.7)

# Print the generated response
print(response.choices[0].message.content.strip())
