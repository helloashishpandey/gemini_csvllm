from flask import Flask, request, jsonify ,render_template
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
# Check if API key is set
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set!")


# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Load the CSV data when the app starts
def load_context():
    loader = CSVLoader(
        file_path="food_items.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"'
        },
    )
    data = loader.load()
    return "\n\n".join([
        f"{doc.page_content}" for doc in data
    ])

context = load_context()

# Initialize the model
model = ChatGoogleGenerativeAI(google_api_key=API_KEY,model="gemini-1.5-flash", temperature=0.3)

# Define the prompt
prompt_template = """
Answer the question as precisely as possible based on the provided context. If the context contains no relevant information or if the input is a greeting like "Hi" or "Hello," respond appropriately.

For greetings such as "Hi" or "Hello," respond with:
"Discussion: Hello! How can I assist you today?"
"Image URL: Not applicable"

If the context contains no relevant information, respond with:
"Discussion: Oh stree kal aana."
"Image URL: Not applicable"

Format your answer as follows:
Discussion: <Your response based on the context or the fallback phrase>
Image URL: <Provide an appropriate image URL or leave a placeholder if unavailable>

Context: 
{context}

Question: 
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Chain setup
chain = LLMChain(llm=model, prompt=prompt)

def parse_response(raw_text):
    # Split the raw response text into lines
    lines = raw_text.split("\n")
    description = "No description available."
    image_url = ""

    # Extract the description and image URL
    for line in lines:
        if line.startswith("Discussion:"):
            description = line.replace("Discussion:", "").strip()
        elif line.startswith("Image URL:"):
            image_url = line.replace("Image URL:", "").strip()

    return description, image_url

@app.route("/", methods=["GET"])
def info():
    # Pass data to the template
    return render_template("index.html", msg="papa hoon me papa")

# Define the API route
@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        # Get the question from the request
        user_input = request.json.get("query")
        print(user_input)
        
        # If no question is provided, return an error
        if not user_input:
            return jsonify({"error": "Question not provided"}), 400
        
        # Generate response using the chain
        response = chain.invoke({"context": context, "question": user_input}, return_only_outputs=True)
        print(response)
        
        raw_text = response.get("text", "")
        description, image_url = parse_response(raw_text)
        # Return the response as JSON

        return jsonify({"description": description, "image": image_url})
        # return jsonify({"response": response})
    
    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
