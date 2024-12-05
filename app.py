from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification

# Flask 초기화
app = Flask(
    __name__, 
    template_folder="../FE/templates",  # HTML 폴더 경로
    static_folder="../FE/static"        # 정적 파일 경로 (CSS, JS 등)
)
CORS(app)

# KeyBERT 모델 초기화
kw_model = KeyBERT()

# BERT 모델 초기화 (문장 유사도 계산용)
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 가벼운 BERT 모델

# Google Custom Search API 설정
API_KEY = "AIzaSyAk_I4aQfzfPFqfaUMu3s3yGGMH826r86M"  # 실제 API 키
SEARCH_ENGINE_ID = "50c1f019089e446d1"  # 실제 검색 엔진 ID

# 양성/음성 단어 리스트 로드
sentiment_words_file = "AI/sentiment_words.csv"  # CSV 파일 경로
data = pd.read_csv(sentiment_words_file)
positive_words = data[data["감정"] == "양성"]["단어"].tolist()
negative_words = data[data["감정"] == "음성"]["단어"].tolist()

# 임계값 설정
threshold_count = 1  # 양성과 음성을 판별할 최소 단어 개수

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

# 키워드 추출 함수 (문장형 키워드 반환)
def extract_keywords_as_single_phrase(text):
    clean_text = preprocess_text(text)
    keybert_keywords = kw_model.extract_keywords(
        clean_text,
        keyphrase_ngram_range=(3, 5),  # 3~5 단어 묶음으로 추출
        stop_words=None,
        top_n=5  # 상위 5개 추출
    )
    phrases = [kw[0] for kw in keybert_keywords]
    return " ".join(phrases)

# BERT 문장 유사도 계산 함수
def calculate_bert_similarity(original_text, article_text):
    original_embedding = bert_model.encode(original_text, convert_to_tensor=True)
    article_embedding = bert_model.encode(article_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(original_embedding, article_embedding)
    return round(float(similarity) * 100, 2)  # 퍼센트로 반환

# 양성/음성 판별 함수
def classify_sentiment(text, positive_words, negative_words, threshold_count):
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    if positive_count >= threshold_count:
        return "양성"
    elif negative_count >= threshold_count:
        return "음성"
    return ""

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
    except requests.RequestException as e:
        return "크롤링 실패"

# 유사 기사 검색 함수
def search_similar_articles(keywords):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,  # Google API 키
        "cx": SEARCH_ENGINE_ID,  # 검색 엔진 ID
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
        text = crawl_text_from_url(url)
        original_keywords = extract_keywords_as_single_phrase(text)
        sentiment_analysis = classify_sentiment(text, positive_words, negative_words, threshold_count)

        # 유사 기사 검색
        similar_articles = search_similar_articles(original_keywords)

        # 유사 기사 정리 및 신뢰도 계산
        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                article["credibility_score"] = calculate_bert_similarity(text, article_text)
            except Exception:
                article["credibility_score"] = 0.0

        # 신뢰도가 높은 순서로 정렬
        similar_articles.sort(key=lambda x: x["credibility_score"], reverse=True)

        # 전체 신뢰도 계산 (유사 기사 신뢰도의 평균)
        overall_credibility = sum(article["credibility_score"] for article in similar_articles) / max(len(similar_articles), 1)

        # 결과 반환
        return jsonify({
            "original_keywords": original_keywords,
            "sentiment_analysis": sentiment_analysis,
            "credibility_score": f"{round(overall_credibility, 2)}%",
            "similar_articles": [
                {
                    "credibility_score": f"{article['credibility_score']}%",
                    "title": article["title"],
                    "url": article["url"]
                }
                for article in similar_articles[:4]  # 상위 2개만 반환
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# HTML 렌더링 엔드포인트
@app.route('/')
def index():
    return render_template('index.html')  # index.html 로드

if __name__ == '__main__':
    app.run(debug=True)
