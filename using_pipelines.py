from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I'vs been waiting for a huggingface course my whole life!!!")

print(res)



# result:
# [{'label': 'POSITIVE', 'score': 0.919508695602417}]