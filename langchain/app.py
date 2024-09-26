from flask import Flask, request, jsonify
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from mysql.connector import Error
from flask_caching import Cache
import mysql.connector
import os
import asyncio

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 캐시 설정 (예: 메모리 캐시 사용)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# 캐시된 AI 응답 함수
@cache.cached(timeout=300, key_prefix='ai_response')
def cached_ai_response(region, radius, question):
    return invoke_chain(region, radius, question)

# 비동기 AI 응답 처리
async def async_invoke_chain(region, radius, question):
    result = await asyncio.to_thread(invoke_chain, region, radius, question)
    return result

# 언어 모델 초기화, 나중에 gpt-4로 변경 예정
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0, max_tokens=1000)

# 메모리 초기화
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI that recommends neighborhoods. Respond with at least three neighborhoods in the format '동네: 설명'. At the end, provide the main keywords from the user's question, prefixed with 'Keywords:'."),
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

# 메모리 로드 함수
def load_memory(input):
    return memory.load_memory_variables({})["history"]

# RunnablePassthrough 체인 정의  
chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

# AI 체인 호출 및 추천 수행 함수
def invoke_chain(region, radius, question):
    # AI 모델에 질문과 지역 정보를 함께 전달하여 처리
    formatted_question = f"지역: {region}, 반경: {radius}km, 질문: {question}"
    result = chain.invoke(
        {
            "question": formatted_question
        }
    )
    memory.save_context(
        {"input": formatted_question},
        {"output": result.content},
    )
    
    # '동네: 설명' 형식으로 추천 결과를 파싱하고, 키워드 추출
    neighborhood_recommendations = []
    keywords = []
    lines = result.content.split('\n')  # 결과를 줄 단위로 분리
    keyword_section = False  # 키워드 구분 플래그
    
    for line in lines:
        line = line.strip()  # 공백 제거
        if not line:
            continue  # 빈 줄은 무시
        if line.startswith('Keywords:'):  # 키워드 섹션 시작
            keyword_section = True
            keywords = line.replace('Keywords:', '').strip().split(',')
            keywords = [keyword.strip() for keyword in keywords]
        elif not keyword_section and ':' in line:  # 키워드 섹션이 아닌 경우에만 동네 정보 파싱
            neighborhood, description = line.split(':', 1)
            neighborhood_recommendations.append({
                "neighborhood": neighborhood.strip(),
                "description": description.strip()
            })
    
    return result.content, neighborhood_recommendations, keywords




# 데이터베이스 초기화 함수
def init_db():
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password=os.getenv("DB_PASSWORD"),
            database='introduceOurTown'
        )
        cursor = conn.cursor()
        # query 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS introduceOurTown.query (
                id INT AUTO_INCREMENT PRIMARY KEY,
                region VARCHAR(255) NOT NULL,
                radius INT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        ''')
        # neighborhood_recommendations 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS introduceOurTown.neighborhood_recommendations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query_id INT NOT NULL,
                neighborhood VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES introduceOurTown.query(id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error: {e}")

# 데이터베이스에 질문과 답변 저장 함수
def save_to_db(region, radius, question, answer, neighborhood_recommendations):
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password=os.getenv("DB_PASSWORD"),
            database='introduceOurTown'
        )
        cursor = conn.cursor()
        # query 테이블에 저장
        cursor.execute(
            'INSERT INTO introduceOurTown.query (region, radius, question, answer) VALUES (%s, %s, %s, %s)',
            (region, radius, question, answer)
        )
        query_id = cursor.lastrowid  # 방금 삽입한 query의 id 가져오기
        
        # neighborhood_recommendations 테이블에 저장
        for neighborhood in neighborhood_recommendations:
            cursor.execute(
                'INSERT INTO introduceOurTown.neighborhood_recommendations (query_id, neighborhood, description) VALUES (%s, %s, %s)',
                (query_id, neighborhood['neighborhood'], neighborhood['description'])
            )
        
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error: {e}")

# Spring 서버에서 JSON 요청을 받음
@app.route('/api/recommend-neighborhoods', methods=['POST'])
def recommend_neighborhoods():
    data = request.get_json()

    # 필수 데이터 확인
    if not data or 'region' not in data or 'radius' not in data or 'question' not in data:
        return jsonify({"error": "Invalid input: 'region', 'radius', and 'question' fields are required"}), 400
    
    region = data['region']
    radius = data['radius']
    question = data['question']
    
    # AI 체인 호출 및 추천 결과 처리
    answer, neighborhood_recommendations, keywords = invoke_chain(region, radius, question)

    # 데이터베이스에 질문과 답변 저장
    save_to_db(region, radius, question, answer, neighborhood_recommendations)

    # 응답 JSON 생성
    response = {
        "region": region,
        "radius": radius,
        "question": question,
        "answer": answer,
        "keywords": keywords,  # 추출된 키워드 추가
        "neighborhood_recommendations": neighborhood_recommendations  # 동네: 설명 형식으로 반환
    }
    
    # 결과를 JSON으로 반환
    return jsonify(response), 200

# Flask 서버 실행
if __name__ == '__main__':
    init_db()  # 데이터베이스 초기화
    app.run('0.0.0.0', port=5001, debug=True)
