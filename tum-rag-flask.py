import os
import openai
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from indexing import initialize_vectorstores

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in the .env file.")
openai.api_key = openai_api_key

# Set up Flask
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize the vector stores (from the previous logic in `indexing.py`)
vectorstores = None

# Initialize the vector stores
current_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_folder = os.path.join(current_dir, "knowledge")
vectorstores = initialize_vectorstores(knowledge_folder, openai_api_key=openai_api_key)

# Helper function to format the context from retrieved documents
def format_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_context_list(docs):
    return [doc.page_content for doc in docs]

# Function to handle the retrieval based on classification
def get_relevant_context(question, classification):
    """
    Retrieves relevant context from the vector stores based on the classification.
    """
    general_retriever = vectorstores["general"].as_retriever(search_kwargs={"k": 5})
    general_docs = general_retriever.get_relevant_documents(question)
    general_context = format_context(general_docs)
    general_context_list = get_context_list(general_docs)
    specific_context = ""
    specific_context_list = []
    
    if classification != "general":
        # Fetch from the specific vector store
        specific_retriever = vectorstores[classification].as_retriever(search_kwargs={"k": 5})
        specific_docs = specific_retriever.get_relevant_documents(question)
        specific_context = format_context(specific_docs)
        specific_context_list = get_context_list(specific_docs)

    return general_context, specific_context, general_context_list, specific_context_list

# Example prompt template that incorporates both general and specific contexts
prompt_template = """You are an assistant helping TUM students with questions about their studies.
Your goal is to provide detailed and accurate answers to their questions.

General information:
{general_context}

Specific information (if available):
{specific_context}

Answer the following question based on only the provided information:
Question: {question}
"""

prompt = PromptTemplate(template=prompt_template)

# Set up LLM for generating answers
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)

# RAG Chain for combining everything together
rag_chain = (
    {"general_context": RunnablePassthrough(),
     "specific_context": RunnablePassthrough(),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Main handler function to classify, retrieve, and answer the question
def handle_question(question, base_folder):
    from study_program_classifier import classify_question  # Import the classify logic
    classification = classify_question(question, base_folder)

    # Get the relevant context based on the classification
    general_context, specific_context, general_context_list, specific_context_list = get_relevant_context(question, classification)

    # Prepare the final prompt
    prompt_text = prompt.format(
        general_context=general_context,
        specific_context=specific_context if specific_context else "No specific context available.",
        question=question
    )

    logging.info(f"Final prompt: {prompt_text}")

    # Generate the answer
    response = llm.invoke(prompt_text)
    return response.content, general_context_list, specific_context_list

# Flask route for handling questions
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('message', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    logging.info(f"Received question: {question}")
    answer, general_context, specific_context = handle_question(question, current_dir)
    logging.info(f"Generated answer: {answer}")
    return jsonify({"answer": answer, "general_context": general_context, "specific_context": specific_context})

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5200)