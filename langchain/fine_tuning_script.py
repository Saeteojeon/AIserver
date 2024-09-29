import openai
import json
from dotenv import load_dotenv
# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# JSONL 파일을 업로드하는 새로운 방식
with open("fine_tune_data.jsonl", "rb") as file:
    response = openai.File.create(
        file=file,
        purpose="fine-tune"
    )
    
# 업로드된 파일의 ID 출력
file_id = response["id"]
print(f"File ID: {file_id}")

# Fine-Tuning 작업 생성 (최신 방식)
fine_tune_response = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-4o-mini"
)

# Fine-Tune 작업 ID 출력
fine_tune_id = fine_tune_response["id"]
print(f"Fine-Tune Job ID: {fine_tune_id}")

# Fine-Tune 작업 상태 확인
status = openai.FineTuningJob.retrieve(fine_tune_id)
print(f"Fine-Tune Status: {status['status']}")
