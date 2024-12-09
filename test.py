from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Flask 초기화
app = Flask(__name__)
CORS(app)

# KeyBERT 모델 초기화
kw_model = KeyBERT()

# 감정 분석 모델 로드
emotion_model_path = "C:/AI-3/kobert_emotion_model.pth"
emotion_model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=7)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
emotion_model.eval()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# BERT 문장 유사도 모델 초기화
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Google Custom Search API 설정
API_KEY = "AIzaSyAk_I4aQfzfPFqfaUMu3s3yGGMH826r86M"
SEARCH_ENGINE_ID = "50c1f019089e446d1"

# 감정 점수 가중치
EMOTION_WEIGHT = 0.2
SIMILARITY_WEIGHT = 0.8

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

# 키워드 추출 함수
def extract_keywords_as_single_phrase(text):
    clean_text = preprocess_text(text)
    keybert_keywords = kw_model.extract_keywords(
        clean_text,
        keyphrase_ngram_range=(3, 5),
        stop_words=None,
        top_n=5
    )
    phrases = [kw[0] for kw in keybert_keywords]
    return " ".join(phrases) if phrases else "키워드 없음"

# 감정 분석 함수
def analyze_emotion(text):
    if not text.strip():  # 텍스트가 비어 있으면 기본값 반환
        return "Unknown"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # 감정 클래스 맵핑
    emotion_map = {
        0: "공포",
        1: "놀람",
        2: "분노",
        3: "슬픔",
        4: "중립",
        5: "행복",
        6: "혐오"
    }
    return emotion_map.get(predicted_class, "Unknown")

# BERT 문장 유사도 계산 함수
def calculate_similarity(text1, text2):
    embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return round(similarity_score * 100, 2)

# 신뢰도 계산 함수
def calculate_credibility(emotion, similarity_score):
    emotion_scores = {
        "공포": 30,
        "놀람": 50,
        "분노": 20,
        "슬픔": 25,
        "중립": 40,
        "행복": 60,
        "혐오": 10
    }
    emotion_score = emotion_scores.get(emotion, 0)
    credibility = (EMOTION_WEIGHT * emotion_score) + (SIMILARITY_WEIGHT * similarity_score)
    return round(credibility, 2)

# 크롤링 함수
def crawl_text_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1')
        title_text = title.get_text(strip=True) if title else "제목 없음"
        body = soup.find('div', {'class': 'article-body'})
        body_text = body.get_text(strip=True) if body else "본문 없음"
        return f"{title_text} {body_text}"
    except requests.RequestException:
        return "크롤링 실패"

# 유사 기사 검색 함수
def search_similar_articles(keywords):
    if keywords == "키워드 없음":
        return []

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": keywords,
        "num": 5
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        return [
            {"title": item.get("title", "제목 없음"), "url": item.get("link", "URL 없음")}
            for item in results.get("items", [])
        ]
    except requests.RequestException:
        return []

# API 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url', '')

    if not url or not url.startswith("http"):
        return jsonify({"error": "Invalid or missing URL"}), 400

    try:
        # 원문 분석
        text = crawl_text_from_url(url)
        original_emotion = analyze_emotion(text)
        original_keywords = extract_keywords_as_single_phrase(text)

        # 유사 기사 검색 및 신뢰도 계산
        similar_articles = search_similar_articles(original_keywords)
        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                similarity_score = calculate_similarity(text, article_text)
                article["credibility_score"] = calculate_credibility(original_emotion, similarity_score)
            except Exception:
                article["credibility_score"] = 0.0

        # 전체 신뢰도 계산
        overall_credibility = sum(article.get("credibility_score", 0) for article in similar_articles) / max(len(similar_articles), 1)

        # 결과 반환
        return jsonify({
            "original_emotion": original_emotion,
            "original_keywords": original_keywords,
            "credibility_score": round(overall_credibility, 2),
            "similar_articles": similar_articles[:2]  # 상위 2개만 반환
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
