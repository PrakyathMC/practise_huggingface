from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels = ["education", "politics", "busines"],
)

print(res)


#result
#{'sequence': 'This is a course about Python list comprehension', 'labels': ['education', 'busines', 'politics'], 'scores': [0.8464775681495667, 0.14388425648212433, 0.009638265706598759]}