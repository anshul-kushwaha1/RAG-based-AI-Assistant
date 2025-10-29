
import requests
import os
import json
import numpy as np
import pandas as pd 
import joblib
from sklearn.metrics.pairwise import cosine_similarity
# from read_chuks import creat_embedding 
# from openai import OpenAI
# from config import api_key

# client = OpenAI(api_key=api_key)

def creat_embedding(text_list):
    r= requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    r= requests.post("http://localhost:11434/api/generate",json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream" : False
    })
    response = r.json()
    print(response)
    return response

# def inference_openai(prompt):
#     response = client.responses.create(
#     model="gpt-5",
#     input=prompt
# )
#     return response.output_text


df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
print("Thinking...")
question_embedding= creat_embedding(incoming_query)[0]
# print(question_embedding)


# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
# find similarity of question embedding with other embedding

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])

prompt = f'''  I am teaching web development using sigma web development course. Here are video subtitle chunks containg video title, video number, start time in seconds, the at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient = "records")}
------------------------------------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that perticular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

# response = inference_openai(prompt)


with open("response.txt", "w") as f:
    f.write(response)


# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
# what type of course will it be  where are semantic tags taught in the course
# where  is box model taught