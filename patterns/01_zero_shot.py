import os
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables
_ = load_dotenv(find_dotenv())

# Load the PDF file using LangChain's PyPDFLoader
loader = PyPDFLoader("./docs/MachineLearning-Lecture01.pdf")
pages = loader.load()

# Combine all pages into a single text
text = " ".join(page.page_content for page in pages)

# Define candidate labels for classification
candidate_labels = [
    "Supervised Learning",
    "Unsupervised Learning",
    "Reinforcement Learning",
    "Neural Networks",
    "Statistical Learning",
    "Optimization Techniques",
    "Data Preprocessing",
    "Model Evaluation"
]
# Use GPT-3.5 Turbo for zero-shot classification
prompt = f"""
You are a helpful assistant. Classify the following text into one of the categories: {', '.join(candidate_labels)}.

Text:
{text}

Only provide the most relevant category.
"""

response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[{"role": "user", "content": prompt}],
max_tokens=50)

# Extract and print the classification result
classification = response.choices[0].message.content.strip()
print("Classification Result:", classification)
