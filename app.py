import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import textwrap
import streamlit as st

### Keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

### Establish Pinecone Client and Connection
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('tech-consulting-rag1')

### Establish OpenAi Client
openai.api_key = openai_api_key
client = openai.OpenAI()

### Get embeddings
def get_embeddings(text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

## Get context
def get_contexts(query, embed_model, k):
    query_embeddings = get_embeddings(query, model=embed_model)
    pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
    context = [item['metadata']['text'] for item in pinecone_response['matches']]
    return context, query

### Augmented Prompt
def augmented_query(user_query, embed_model, k=5):
    context, query = get_contexts(user_query, embed_model=embed_model, k=k)
    return "\n\n---\n\n".join(context) + "\n\n---\n\n" + query

### Ask GPT
def ask_gpt(system_prompt, user_prompt, model, temp=0.7):
    temperature_ = temp
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature_,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)

### Education ChatBot
def Education_ChatBot(query):
    embed_model = 'text-embedding-ada-002'
    primer = """
    You are an Education Assistant AI for the "Managing Big Data" course. Your job is to create quizzes and help students understand course content. Use only the provided knowledge base, and if you lack information to answer, say: "I do not know based on the information provided." Do not use external information or make assumptions.

    Create quizzes with ten questions that vary in type: multiple choice, open-ended, and coding tasks. For example, ask, "Which framework is used for distributed computing?" or "Write a SQL query to find the top 5 sales by region." Ensure questions align with the material, and if not possible, respond: "I do not know based on the information provided."

    When students ask questions, give clear answers based on the course. For instance, if a student asks, "What is batch processing?" explain it as covered in the material. If asked beyond the course scope, simply say, "I do not know based on the information provided." Keep responses clear, helpful, and focused on learning.

    Stay within the course's boundaries. Your answers must be accurate, brief, and supportive to ensure an effective learning experience.
    """

    llm_model = 'chatgpt-4o-latest'
    user_prompt = augmented_query(query, embed_model)
    return ask_gpt(primer, user_prompt, model=llm_model)


### Streamlit Interface with Dynamic Input ###

st.title("Education ChatBot")

# Initialize the session state to store questions and answers
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to add a new question and display responses
def add_new_interaction():
    question = st.session_state.new_question
    if question.strip():
        response = Education_ChatBot(question)
        st.session_state.conversation.append((question, response))
    st.session_state.new_question = "" 

# Display past conversations
for idx, (question, response) in enumerate(st.session_state.conversation):
    st.write(f"**Q{idx+1}:** {question}")
    st.write(f"**A{idx+1}:** {response}")

# Ask the first question or follow-up questions
st.text_input("Ask your question here:", key="new_question", on_change=add_new_interaction)