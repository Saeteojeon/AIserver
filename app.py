from flask import Flask, request, render_template_string
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

app = Flask(__name__)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

# Initialize the memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human. You have to recommend suitable neighborhoods when human ask you questions. When you recommend a neighborhood, recommend several neighborhoods and let me know the dong or eup units and explain it. and When you recommend a neighborhood, put a number on it and recommend it"),
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

    # Split the result into lines and process each recommendation
    lines = result.content.split('\n')
    recommendations = []
    for line in lines:
        if line.strip() and line[0].isdigit():
            parts = line.split(' ', 1)
            if len(parts) == 2:
                recommendations.append(parts[1].strip())

    return result.content, recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form['question']
        answer, recommendations = invoke_chain(question)
        return render_template_string(TEMPLATE, question=question, answer=answer, recommendations=recommendations)
    return render_template_string(TEMPLATE)

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
        {% if recommendations %}
        <h2>Recommended Neighborhoods:</h2>
        <ul>
        {% for recommendation in recommendations %}
        <li>{{ recommendation }}</li>
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
    app.run('0.0.0.0', port=5001, debug=True)
