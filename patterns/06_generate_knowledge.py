from openai import OpenAI
import os

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input query for the task
query = "Can patients with G6PD deficiency take aspirin?"

# Step 1: Generate multiple pieces of knowledge
knowledge_prompt = f"""
You are a medical expert. Generate multiple pieces of structured knowledge to help answer the following question. Each knowledge point should focus on a specific aspect of the condition, the medication, or related interactions.

{query}
Knowledge Points:
1.
2.
3.
"""

# Call OpenAI API to generate multiple knowledge points
knowledge_response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates multiple structured knowledge points."},
        {"role": "user", "content": knowledge_prompt}
    ],
    max_tokens=700,
    temperature=0
)

# Extract generated knowledge points
generated_knowledge = knowledge_response.choices[0].message.content.strip()
# print("Generated Knowledge:\n", generated_knowledge)

# Step 2: Integrate the knowledge into reasoning and prediction
reasoning_prompt = f"""
You are a medical reasoning assistant. Use the provided knowledge points to explain "{query}". Your reasoning should integrate all knowledge points to arrive at an accurate and well-supported answer.

Question: {query}
Knowledge Points: {generated_knowledge}
In your answer, don't include the knowledge points. Just give the response.
"""

# Call OpenAI API to reason and predict
reasoning_response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a medical assistant that reasons using multiple knowledge points."},
        {"role": "user", "content": reasoning_prompt}
    ],
    max_tokens=700,
    temperature=0
)

# Extract the final explanation and answer
final_answer = reasoning_response.choices[0].message.content.strip()
print(final_answer)
