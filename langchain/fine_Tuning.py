from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import xml.etree.ElementTree as ET
import requests  # requests 라이브러리 import 추가
from dotenv import load_dotenv
import os

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 환경 변수 로드 (API 키 등)
load_dotenv()

# GPT 모델 설정 (파인튜닝된 모델 사용)
llm = ChatOpenAI(model="ft:gpt-4o-mini-2024-07-18:personal:langchain-v1:ACRn6jO7", temperature=0.5)

# 서울시 공공 데이터 API 호출 함수
def fetch_seoul_data():
    api_key = os.getenv('SEOUL_API_KEY')  # 환경 변수에서 API 키를 불러옴
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/xml/SebcArtGalleryKor/1/5/"
    
    # API 호출
    response = requests.get(url)  # requests 라이브러리로 API 호출
    
    if response.status_code == 200:
        # XML 파싱
        root = ET.fromstring(response.content)
        return root
    else:
        print(f"Error: {response.status_code}")
        return None

# 서울시 API 데이터 파싱 함수
def parse_seoul_data(root):
    data_list = []
    for row in root.findall('.//row'):
        main_key = row.find('MAIN_KEY').text
        category = row.find('CATEGORY').text
        kor_name = row.find('KOR_NAME').text
        kor_gu = row.find('KOR_GU').text
        kor_add = row.find('KOR_ADD').text
        
        # 필요한 정보만 선택하여 리스트에 추가
        data_list.append({
            "main_key": main_key,
            "category": category,
            "kor_name": kor_name,
            "kor_gu": kor_gu,
            "kor_add": kor_add
        })
    return data_list

# 질문에서 관련 키워드 추출 함수
def extract_keywords(question):
    keywords = question.split()  # 간단히 공백으로 분리하여 키워드 추출
    return keywords

# 검색 결과를 필터링하는 함수
def filter_search_results(search_results, keywords):
    filtered_results = []
    for result in search_results:
        if any(keyword in result['kor_name'] or keyword in result['kor_add'] for keyword in keywords):
            filtered_results.append(result)
    return filtered_results

# 데이터를 GPT 모델에 전달할 포맷으로 정리하는 함수
def prepare_data_for_gpt(data_list):
    formatted_data = ""
    for data in data_list:
        formatted_data += f"동네: {data['kor_gu']}, 장소: {data['kor_name']}, 주소: {data['kor_add']}\n"
    return formatted_data

# LLMChain 프롬프트 템플릿 설정
prompt_template = ChatPromptTemplate.from_template(
    """
    The user asked for neighborhood recommendations.
    Here's some reference data from Seoul city public sources:
    {retrieved_info}.
    But, you should primarily base your response on the model's knowledge and training.
    Respond with at least three detailed neighborhoods in the format:
    Location: ~구 ~동
    Description: 관련 설명.
    """
)

# LLMChain 생성
chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

# Flask API 엔드포인트
@app.route('/api/recommend-neighborhoods', methods=['POST'])
def recommend_neighborhoods():
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid input: 'question' field is required"}), 400

    question = data['question']
    
    # 서울시 데이터 가져오기
    root = fetch_seoul_data()
    if root is None:
        return jsonify({"error": "Failed to retrieve Seoul data"}), 500
    
    # 데이터 파싱
    seoul_data = parse_seoul_data(root)
    
    # 질문에서 키워드 추출
    keywords = extract_keywords(question)
    
    # 검색 결과 필터링 (질문과 관련된 데이터만 사용)
    filtered_data = filter_search_results(seoul_data, keywords)
    
    # GPT에 전달할 데이터 준비
    formatted_data = prepare_data_for_gpt(filtered_data)
    
    # GPT 모델 실행 (모델이 데이터를 참고하되, 파인튜닝된 지식을 바탕으로 답변 생성)
    result = chain.run({
        "user_question": question,
        "retrieved_info": formatted_data  # 참고용 데이터로만 제공
    })
    
    # 3개 이상의 동네가 포함된 형식으로 JSON 생성
    json_result = {
        "keywordList": keywords,  # 질문에서 추출한 키워드
        "answer": []
    }

    # GPT 결과를 파싱해서 JSON에 추가
    for line in result.split('\n'):
        if line.startswith("Location"):
            location, description = line.split('Description:', 1)
            json_result["answer"].append({
                "location": location.replace('Location:', '').strip(),
                "description": description.strip()
            })
    
    return jsonify(json_result), 200

# 서버 실행
if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)
