from flask import Flask, request, render_template_string, jsonify
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from mysql.connector import Error
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))
# Initialize the memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human. You have to recommend suitable neighborhoods when human ask you questions. When you recommend a neighborhood, please recommend several neighborhoods and let me know the dong or eup units . And When you talk about the neighborhood, put a number on it and show it to human"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create the LLMChain
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

# Define a function to load memory
def load_memory(input):
    return memory.load_memory_variables({})["history"]
    
# Define the runnable chain with memory assignment  
chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

# Define a function to invoke the chain and save context
def invoke_chain(question):
    result = chain.invoke(
        {
        "question": question
        }
    )
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    # Extract neighborhood names ending with 'dong' or 'eup'
    relevant_neighborhoods = []
    words = result.content.replace(',', ' ').split()  # Replace commas with spaces and then split
    for word in words:
        clean_word = word.strip('.,**:()')
        if clean_word.endswith('-dong') or clean_word.endswith('-eup'):
            relevant_neighborhoods.append(clean_word)

    return result.content, relevant_neighborhoods

# Database setup
def init_db():
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='Tngh1004!!',
            database='introduceOurTown'
        )
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS introduceOurTown.query (
                id INT AUTO_INCREMENT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error: {e}")

def save_to_db(question, answer):
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='Tngh1004!!',
            database='introduceOurTown'
        )
        cursor = conn.cursor()
        cursor.execute('INSERT INTO introduceOurTown.query (query, answer) VALUES (%s, %s)', (question, answer))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error: {e}")

#localhost 연결
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form['question']
        answer, relevant_neighborhoods = invoke_chain(question)
        save_to_db(question=question, answer=answer)
        return render_template_string(TEMPLATE, question=question, answer=answer, relevant_neighborhoods=relevant_neighborhoods)
    return render_template_string(TEMPLATE)

#postman으로 연결
@app.route('/get/gpt_request', methods=['POST'])
def get_gpt_request():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        answer, relevant_neighborhoods = invoke_chain(question)
        save_to_db(question=question, answer=answer)

        response = {
            "message": "Data saved successfully",
            "data": {
                "question": question,
                "answer": answer
            }
        }

        return jsonify(response), 200
    return render_template_string(TEMPLATE)

#localhost 연결하기 위한 TEMPLATE
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Neighborhood Recommender</title>
</head>
<body>
    <div class="container">
    <h1 class="mt-5">Neighborhood Recommender</h1>
    <form method="POST">
        <div class="form-group">
        <label for="question">Ask a question:</label>
        <input type="text" class="form-control" id="question" name="question" required>
        </div>
        <button type="submit" class="btn btn-primary">Submit</button>
    </form>
    {% if question %}
    <div class="mt-5">
        <h2>Question:</h2>
        <p>{{ question }}</p>
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
        {% if relevant_neighborhoods %}
        <h2>recommended neighborhoods:</h2>
        <ul>
        {% for neighborhood in relevant_neighborhoods %}
        <li>{{ neighborhood }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endif %}
    </div>
</body>
</html>
"""


if __name__ == '__main__':
    init_db()
    app.run('0.0.0.0', port=5001, debug=True)
