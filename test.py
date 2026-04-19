from transformers import pipeline

print(pipeline("text2text-generation", model="google/flan-t5-base"))