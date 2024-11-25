from transformers import pipeline
import os

# device=0 initializes the graphics card. Remove this or set to -1 for CPU
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# text = "I'm currently trying to install a graphics card on my laptop, but not for fun."

# Read text from a file
with open('./corpus/lorem_ipsum.txt', 'r') as file:
    text = file.read()

candidate_labels = ["technology", "shopping", "finance", "travel", "entertainment"]

# Perform zero-shot classification
result = classifier(text, candidate_labels)

print("Classification Results:", result)
