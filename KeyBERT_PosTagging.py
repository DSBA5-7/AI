from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from konlpy.tag import Okt
import re

# Flask 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화 (필요한 경우)

# KeyBERT 초기화
kw_model = KeyBERT()

# KoNLPy의 Okt 형태소 분석기 초기화
okt = Okt()

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

# POS tagging을 활용한 주어, 지역, 날짜 추출 함수
def extract_contextual_keywords(text):
    tokens = okt.pos(text, norm=True, stem=True)
    subjects, locations, dates = [], [], []
    
    for i, (word, tag) in enumerate(tokens):
        # 주어 추출
        if tag == "Noun" and i + 1 < len(tokens) and tokens[i + 1][1] == "Josa" and tokens[i + 1][0] in ["이", "가", "은", "는"]:
            subjects.append(word)
        # 지역 추출
        elif tag == "Noun" and i + 1 < len(tokens) and tokens[i + 1][1] == "Josa" and tokens[i + 1][0] in ["에서", "에"]:
            locations.append(word)
        # 날짜 추출
        elif re.match(r"^\d+월|\d+일|\d+년$", word):
            dates.append(word)
    
    return list(set(subjects)), list(set(locations)), list(set(dates))

# 키워드 추출 함수
def extract_keywords_with_pos(text):
    clean_text = preprocess_text(text)

    # KeyBERT 키워드 추출
    keybert_keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1, 1), top_n=10)
    keybert_keywords = [kw[0] for kw in keybert_keywords]

    # POS tagging으로 주어, 지역, 날짜 추출
    subjects, locations, dates = extract_contextual_keywords(clean_text)

    # 키워드 통합
    combined_keywords = list(set(keybert_keywords + subjects + locations + dates))
    return combined_keywords

# Flask API 엔드포인트
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 키워드 추출
    keywords = extract_keywords_with_pos(text)
    return jsonify({"keywords": keywords})

# Flask 실행
if __name__ == '__main__':
    app.run(debug=True)