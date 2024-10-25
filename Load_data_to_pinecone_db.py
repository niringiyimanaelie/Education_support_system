"""Libraries"""

import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import numpy as np
import pdfplumber


### Keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

### Establish Pinecone Client and Connection
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('managingbigdata')

### Establish OpenAi Client
openai.api_key = openai_api_key
client = openai.OpenAI()

"""Embedding function"""

def get_embeddings(text, model='text-embedding-ada-002'):
  text = text.replace("\n", " ")
  return client.embeddings.create(input = text, model=model).data[0].embedding

"""Dataset"""

# filepaths = "Managing_bigdata_merged_notes.pdf"
# with open(filepaths, encoding='utf-8', errors='ignore') as f:
#     data = f.read() + "\n"
def read_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Use the new function to extract text from the PDF
filepaths = "Managing_bigdata_merged_notes.pdf"
data = read_pdf(filepaths) 


"""Split"""

my_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100,
    length_function = len
)

chunks = my_splitter.split_text(data)

schema = pd.DataFrame(columns=['id', 'values', 'metadata'])

def generate_ids(number, size):
  import string, random
  ids=[]
  for i in range(number):
    res = ''.join(random.choices(string.ascii_letters,k=size))
    ids.append(res)
    if len(set(ids)) != i+1:
      i-=1
      ids.pop(-1)

  return ids

def load_chunks(df, split_text):
  ids = generate_ids(len(split_text), 7)
  i = 0
  for chunk in split_text:
    df.loc[i] = [ids[i], get_embeddings(chunk), {'text': chunk}]
    i+=1
  return df

pinecone_data = load_chunks(schema, chunks)

pinecone_data.to_csv('Managing_bigdata_merged_notes.csv')
pinecone_df= pd.read_csv('Managing_bigdata_merged_notes.csv')

def prepare_DF(df):
  import json,ast
  try: df=df.drop('Unnamed: 0',axis=1)
  except: print('Unnamed Not Found')
  df['values']=df['values'].apply(lambda x: np.array([float(i) for i in x.replace("[",'').replace("]",'').split(',')]))
  df['metadata']=df['metadata'].apply(lambda x: ast.literal_eval(x))
  return df

index_df = prepare_DF(pinecone_df)

def chunker(seq, size):
 'Yields a series of slices of the original iterable, up to the limit of what size is.'
 for pos in range(0, len(seq), size):
   yield seq.iloc[pos:pos + size]

def convert_data(chunk):
 'Converts a pandas dataframe to be a simple list of tuples, formatted how the `upsert()` method in the Pinecone Python client expects.'
 data = []
 for i in chunk.to_dict('records'):
  data.append(i)
 return data

for chunk in chunker(index_df, 200):
  index.upsert(vectors=convert_data(chunk))

print("Loading data completed successfully.!!!!!")

