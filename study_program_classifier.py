import os
import openai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def load_study_programs(file_path):
    """
    Load the study programs from study_programs.txt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Study programs file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        study_programs = [line.strip() for line in f.readlines() if line.strip()]
    
    return study_programs

def ask_llm_to_classify(question, study_programs):
    """
    Use the OpenAI Chat Completions API to determine if the question is study program-specific and, if so, which study program.
    """
    study_programs_str = ", ".join(study_programs)

    # Define the system and user messages for the chat completion API
    messages = [
        {"role": "system", "content": "You are an assistant that classifies questions based on study programs."},
        {"role": "user", "content": f"""
        Given the following question: "{question}", please determine if it is specific to any of the following study programs:
        {study_programs_str}.
        
        If it is specific to one of the study programs, respond with the name of the program. If the student studies Master Wirtschaftsinformatik, respond with Master Information Systems. If it is general and not specific to any, respond with 'general'.
        """}
    ]

    # Send the request to the OpenAI Chat Completions API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use the desired model
        messages=messages,    # Use the new messages format
        max_tokens=50,        # Limit tokens to avoid overly verbose responses
        temperature=0.7       # Adjust creativity level (0 = deterministic, 1 = more creative)
    )

    # Extract the assistant's message from the response
    response_message = response.choices[0].message.content.strip()
    print(f"CLASSIFICATION RESPONSE: {response_message}")

    return response_message


def format_as_folder_name(study_program_name):
    """
    Convert human-readable study program name to a folder-friendly format.
    This involves converting to lowercase and replacing spaces with hyphens.
    """
    return study_program_name.lower().replace(" ", "-")

def classify_question(question, base_folder):
    """
    Classifies the question as either 'general' or study-program-specific based on the study programs list
    in the study_programs.txt file.
    """
    # Path to the study_programs.txt file
    study_programs_file = os.path.join(base_folder, "study_programs.txt")

    # Load study programs from the study_programs.txt file
    study_programs = load_study_programs(study_programs_file)

    # Ask ChatGPT to classify the question
    result = ask_llm_to_classify(question, study_programs)

    # If ChatGPT says 'general', return 'general'
    if result.lower() == "general":
        return "general"
    
    # Convert the result to a folder name
    folder_name = format_as_folder_name(result)
    
    return folder_name