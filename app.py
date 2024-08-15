import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# prompt_template = """
# You are a car manual expert and highly knowledgeable about vehicle maintenance and operation.
# Based on the following excerpt from a car manual, diagrams, and tables:
# {context}
# Question: {question}
# If the question is a general greeting or a non-vehicle related statement, respond with "Hello! How can I assist you with vehicle maintenance today?"
# If the information is not available or the question is outside the context of vehicle maintenance, respond with "Sorry, I don't have much information about it."
# Provide a detailed and informative answer based on the context provided if the question is related to vehicle maintenance and operation.
# Answer:
# """

prompt_template = """
You are a car manual expert and highly knowledgeable about vehicle maintenance and operation.
Based on the following excerpt from a car manual, diagrams, and tables:
{context}
Question: {question}
If the question is a general greeting or any type of greetings, respond with "Hello, How can I assist you on vehicle maintenance."
If the question is not a greeting do not add "Hello, How can I assist you on vehicle maintenance."
If the information is not available in the manual but is related to cars/vehicles, provide a detailed and informative answer. Include images if necessary.
For long answers, present the information in bullet points with side headings neatly presented.
If the question is outside the context of vehicle maintenance, respond with "Sorry, I don't have much information about it."
If the question is related to the PDF content, provide the answer strictly from the PDF without any extra information.
Answer:
"""


# Initialize the QA chain with the prompt template and OpenAI model
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, max_tokens=1024)

# Load the FAISS index and metadata using langchain_community.vectorstores.FAISS
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
print("FAISS index loaded.")

def is_car_related(question):
    car_keywords = ['car', 'vehicle', 'automobile', 'engine', 'transmission', 'brakes', 'tires', 'maintenance', 'repair', 'oil change', 'fuel', 'dashboard']
    return any(keyword in question.lower() for keyword in car_keywords)

def answer(question):
    print(f"Question: {question}")
    
    # Log the type and contents of vectorstore
    print(f"Vectorstore type: {type(vectorstore)}")
    
    # Perform similarity search
    try:
        relevant_docs = vectorstore.similarity_search(question)
        print(f"Relevant Docs: {relevant_docs}")
    except Exception as e:
        print(f"Error during similarity search: {e}")
    
    if not relevant_docs:
        print("No relevant documents found.")
    
    context = ""
    relevant_images = []
    for d in relevant_docs:
        print(f"Document: {d}")
        if d.metadata.get('type') == 'text':
            context += '[text]' + d.metadata.get('original_content', '')
        elif d.metadata.get('type') == 'table':
            context += '[table]' + d.metadata.get('original_content', '')
        elif d.metadata.get('type') == 'image':
            context += '[image]' + d.page_content
            image_data = d.metadata.get('original_content', '')
            if image_data:
                print(f"Encoding image data for {d.metadata}")
                relevant_images.append(image_data)
    
    print(f"Context: {context}")
    result = llm(prompt.format(context=context, question=question))
    print(f"Result: {result}")
    result_text = result.content if hasattr(result, 'content') else str(result)
    # Determine the response based on the result_text
    if "Sorry, I don't have much information about it." in result_text:
        return result_text, []
    if "Hello, How can I assist you on vehicle maintenance?" in result_text:
        return result_text, []
    return result_text, relevant_images

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    print(f"Received Question: {question}")  # Log the received question
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Temporarily bypass car-related keyword filtering for testing
    # if is_car_related(question):
    result, relevant_images = answer(question)
    response = {'answer': result}
    
    if relevant_images:
        print(f"Adding images to response: {relevant_images}")  # Log images being added to response
        response['images'] = relevant_images
    
    print(f"Response: {response}")  # Log the response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
