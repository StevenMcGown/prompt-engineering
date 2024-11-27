from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in the environment variables.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

url = "https://en.wikipedia.org/wiki/Quantum_mechanics"
question = "What are the core tenets of quantum mechanics?"
loader = WebBaseLoader(url)

try:
    documents = loader.load()
except Exception as e:
    print(f"Error loading webpage: {e}")
    documents = []

if not documents:
    raise ValueError("No documents were loaded from the webpage.")

# Combine the document content into a single text
webpage_content = " ".join([doc.page_content for doc in documents])

# Define maximum tokens for splitting
max_chunk_tokens = 2000
chunks = [webpage_content[i:i + max_chunk_tokens] for i in range(0, len(webpage_content), max_chunk_tokens)]

# Summarize each chunk to reduce size
summaries = []
for idx, chunk in enumerate(chunks):
    summarize_prompt = f"""
    Please summarize the following text in 300 words or less. Focus on key points relevant to the topic.
    ####
    {chunk}
    ####
    """
    try:
        response = llm.invoke(summarize_prompt)
        summaries.append(response.content.strip())
    except Exception as e:
        print(f"Error summarizing chunk {idx + 1}: {e}")
        continue

# Combine summaries for the final document content
summarized_content = " ".join(summaries)

# Question to answer

# Prompt 1: Extract Relevant Quotes
quote_extraction_prompt = f"""
Extract quotes relevant to answering the question from the document below.
Provide the quotes inside <quotes></quotes>. Respond with 'No relevant quotes found!' if no relevant quotes are found.
####
{summarized_content}
####
Question: {question}
"""

try:
    quote_response = llm.invoke(quote_extraction_prompt)
    extracted_quotes = quote_response.content.strip()
except Exception as e:
    print(f"Error extracting quotes: {e}")
    extracted_quotes = "No relevant quotes found!"

# Prompt 2: Generate Answer Using Extracted Quotes
if extracted_quotes != "No relevant quotes found!":
    answer_generation_prompt = f"""
    Given the relevant quotes (delimited by <quotes></quotes>) extracted from the document, and the question below,
    compose a helpful and accurate answer in a friendly tone.
    ####
    {summarized_content}
    ####
    <quotes>
    {extracted_quotes}
    </quotes>
    Question: {question}
    """
    try:
        answer_response = llm.invoke(answer_generation_prompt)
        final_answer = answer_response.content.strip()
        print(final_answer)
    except Exception as e:
        print(f"Error generating answer: {e}")
else:
    print("No relevant quotes found to generate an answer.")
