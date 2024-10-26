# Education ChatBot for Managing Big Data
# Overview
This project is an AI-powered chatbot designed to enhance learning in the "Managing Big Data" course. Professors can use it to create quizzes, while students can use it to learn and interact with the course's knowledge base. The system leverages OpenAI and Pinecone to provide context-based responses and quiz generation.

# Features
Extracts and embeds text from PDFs for course content.
Generates diverse quiz questions based on course material.
Provides accurate, context-based answers to student queries.

# Installation
# Clone the repository:
https://github.com/niringiyimanaelie/Education_support_system.git
git clone https://github.com/niringiyimanaelie/Education_support_system.git
cd Education_support_system

# Install dependencies:
pip install -r requirements.txt

# Set up environment variables: Create a .env file with:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Usage
# Run the Streamlit app:
streamlit run <app.py>