import os
from flask import Flask, request, jsonify
from google.cloud import vision
from google.oauth2 import service_account
import requests

app = Flask(__name__)

# Google Places API 키 (환경 변수으로 설정 권장)
GOOGLE_PLACES_API_KEY = os.getenv('VISION_API_KEY')



# Google Cloud Vision 클라이언트 설정
credentials = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
)
client = vision.ImageAnnotatorClient(credentials=credentials)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        content = image_file.read()
        image = vision.Image(content=content)
        
        # 이미지 라벨 감지
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        if response.error.message:
            raise Exception(response.error.message)
        
        # 라벨 목록 추출
        label_descriptions = [label.description for label in labels]
        
        # 예시: 첫 5개의 라벨을 사용하여 장소 검색
        query = ' '.join(label_descriptions[:5])
        
        # Google Places API를 사용하여 유사한 장소 검색
        places_url = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
        params = {
            'query': query,
            'key': GOOGLE_PLACES_API_KEY
        }
        
        places_response = requests.get(places_url, params=params)
        places_data = places_response.json()
        
        if places_response.status_code != 200:
            return jsonify({'error': 'Error fetching places data'}), 500
        
        # 장소 정보 추출
        places = []
        for place in places_data.get('results', []):
            places.append({
                'name': place.get('name'),
                'address': place.get('formatted_address'),
                'rating': place.get('rating'),
                'types': place.get('types')
            })
        
        return jsonify({
            'labels': label_descriptions,
            'similar_places': places
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 환경 변수로 GOOGLE_APPLICATION_CREDENTIALS 설정 (옵션)
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_FILE
    app.run(host='0.0.0.0', port=5000, debug=True)
