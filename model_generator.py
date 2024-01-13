from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course you will see how to",
    max_length = 30,
    num_return_sequences = 2,

)

print(res)


#results
# [{'generated_text': 'In this course you will see how to make an impact. I think it is a smart idea to teach and teach others about the strengths and benefits of'}, {'generated_text': 'In this course you will see how to change the way you think about the way things work. You will see how it looks in software, how it'}]