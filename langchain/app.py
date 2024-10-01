from flask import Flask, request, jsonify
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from flask_swagger_ui import get_swaggerui_blueprint
from flask_caching import Cache
from mysql.connector import Error
import mysql.connector
import os

from dotenv import load_dotenv

app =  Flask(__name__)


# 환경 변수 로드
load_dotenv()

# 언어 모델 초기화, 나중에 gpt-4로 변경 예정
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, max_tokens=1000, )


# 캐시 설정 (메모리 캐시 사용)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# 캐시된 AI 응답 함수
@cache.cached(timeout=300, key_prefix='ai_response')
def cached_invoke_chain(question):
    return invoke_chain(question)

# 메모리 초기화
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI that recommends neighborhoods. Your response should follow this format:\nKeyword: [list of keywords]\nLocation: [recommended area]\ndescription: [description of the area].\nAnswer with at least two neighborhoods."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# LLMChain 생성
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

# Swagger 설정
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Swagger 정의 파일 경로

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Neighborhood Recommendation API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def invoke_chain(question):
    # AI 모델에 질문을 전달하여 처리
    result = chain.invoke(
        {
            "question": question
        }
    )
    
    # result가 dict 형식으로 반환되었을 때, 'text' 키에서 응답을 가져옴
    response_text = result["text"] if isinstance(result, dict) else result

    # 디버깅용으로 응답 출력
    print(f"AI 응답: {response_text}")

    memory.save_context(
        {"input": question},
        {"output": response_text},
    )
    
    # 응답을 파싱하여 키워드 및 동네 정보 추출
    keywords = []
    neighborhood_recommendations = []
    lines = response_text.split('\n')  # 결과를 줄 단위로 분리
    
    current_location = ""
    current_description = ""
    location_description_map = []  # Location과 description을 매핑할 리스트
    
    for line in lines:
        line = line.strip()
        if line.startswith("Keyword:"):
            # Keyword: 이후의 내용을 파싱하여 리스트로 저장
            keywords = [kw.strip() for kw in line.replace("Keyword:", "").split(',')]
        elif line.startswith("Location:"):
            # Location에 여러 개의 장소가 있을 수 있으므로 분리
            locations = line.replace("Location:", "").strip().split(',')
        elif line.startswith("description:") or line.startswith("Description:"):
            # description을 추출하여 임시 저장
            current_description = line.replace("description:", "").replace("Description:", "").strip()
            # 각각의 location에 description을 할당
            for location in locations:
                neighborhood_recommendations.append({
                    "location": location.strip(),
                    "description": current_description
                })
    
    return response_text, neighborhood_recommendations, keywords

# Flask 서버에서 JSON 요청을 받음
@app.route('/api/recommend-neighborhoods', methods=['POST'])
def recommend_neighborhoods():
    try:
        # spring 서버로부터 요청 데이터 받기
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "Invalid input: 'question' field is required"}), 400
        
        question = data['question']
        
        # AI 체인 호출 및 추천 결과 처리
        answer, neighborhood_recommendations, keywords = invoke_chain(question)

        # 캐시에서 AI 체인 호출 및 추천 결과 처리
        answer, neighborhood_recommendations, keywords = cached_invoke_chain(question)
        
        # 응답 JSON 생성
        response = {
            "keywordList": keywords,  # 추출된 키워드 추가
            "answer": [  # 여러 개의 동네 추천을 리스트로 반환
                {
                    "location": rec["location"],
                    "description": rec["description"]
                } for rec in neighborhood_recommendations
            ]
        }
        
        # 결과를 JSON으로 반환
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

'''
# 데이터베이스 연결 함수
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='your_password',
            database='recommendation_db'
        )
        if conn.is_connected():
            print("Connected to the database")
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# 데이터베이스에 질문과 추천 결과 저장
def save_to_db(question, keywords, recommendations):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        # query 테이블에 질문과 키워드 저장
        cursor.execute('INSERT INTO query (question, keywords) VALUES (%s, %s)', (question, ','.join(keywords)))
        query_id = cursor.lastrowid
        
        # recommendation 테이블에 각 추천 결과 저장
        for rec in recommendations:
            cursor.execute(
                'INSERT INTO recommendation (query_id, location, description) VALUES (%s, %s, %s)',
                (query_id, rec["location"], rec["description"])
            )
        
        conn.commit()
        cursor.close()
        conn.close()
'''
# Flask 서버 실행
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
