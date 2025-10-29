import requests
import os
import json
import numpy as np
import pandas as pd 
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
def creat_embedding(text_list):
    r= requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding



jsons = os.listdir("jsons") #list all the jsons 

my_dicts =[]
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = creat_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        # print(chunk)
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        # if(i==5):  # read 5 chunks
        #     break 

    # break
# print(my_dicts)
df = pd.DataFrame.from_records(my_dicts)

# save this data frame
joblib.dump(df, 'embeddings.joblib')








# print(df)
# incoming_query = input("Ask a Question: ")
# question_embedding= creat_embedding(incoming_query)[0]
# print(question_embedding)


# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
# find similarity of question embedding with other embedding

# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
# new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])


# what type of course will it be