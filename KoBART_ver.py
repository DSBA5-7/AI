from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BartForConditionalGeneration, BartTokenizer

# Flask 초기화
app = Flask(
    __name__, 
    template_folder="../FE/templates",  # HTML 폴더 경로
    static_folder="../FE/static"        # 정적 파일 경로
)
CORS(app)

# KoBART 모델 및 토크나이저 로드
model_path = "/Desktop/KoBART-summarization/final_model"  # 파인튜닝된 KoBART 모델 경로
kobart_model = BartForConditionalGeneration.from_pretrained(model_path)
kobart_tokenizer = BartTokenizer.from_pretrained(model_path)

# BERT 문장 유사도 모델 초기화
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Google Custom Search API 설정
API_KEY = "AIzaSyAk_I4aQfzfPFqfaUMu3s3yGGMH826r86M"
SEARCH_ENGINE_ID = "50c1f019089e446d1"

# CSV 파일 기반 양성/음성 단어 로드
sentiment_words_file = "sentiment_words.csv"  # CSV 파일 경로
data = pd.read_csv(sentiment_words_file)
positive_words = data[data["감정"] == "양성"]["단어"].tolist()
negative_words = data[data["감정"] == "음성"]["단어"].tolist()

# 감정 점수 가중치
EMOTION_WEIGHT = 0.2
SIMILARITY_WEIGHT = 0.7
SENTIMENT_WEIGHT = 0.1

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

# KoBART 기반 키워드 추출 (요약)
def extract_keywords_as_single_phrase(text):
    inputs = kobart_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = kobart_model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0, num_beams=4)
    summary = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 양성/음성 분석 함수
def classify_sentiment(text, positive_words, negative_words):
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        return "양성"
    elif negative_count > positive_count:
        return "음성"
    return "중립"

# BERT 문장 유사도 계산 함수
def calculate_similarity(text1, text2):
    embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return round(similarity_score * 100, 2)

# 신뢰도 계산 함수
def calculate_credibility(similarity_score, sentiment):
    sentiment_scores = {"양성": 40, "음성": 40, "중립": 20}
    sentiment_score = sentiment_scores.get(sentiment, 30)
    credibility = SIMILARITY_WEIGHT * similarity_score + SENTIMENT_WEIGHT * sentiment_score
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
        text = crawl_text_from_url(url)
        original_keywords = extract_keywords_as_single_phrase(text)
        original_sentiment = classify_sentiment(text, positive_words, negative_words)

        similar_articles = search_similar_articles(original_keywords)
        if not similar_articles:
            return jsonify({
                "original_sentiment": original_sentiment,
                "original_keywords": original_keywords,
                "credibility_score": "측정 불가",
                "similar_articles": []
            })

        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                similarity_score = calculate_similarity(text, article_text)
                credibility_score = calculate_credibility(similarity_score, original_sentiment)
                article["credibility_score"] = f"{credibility_score}%"
            except Exception:
                article["credibility_score"] = "측정 불가"

        # 신뢰도 높은 순으로 정렬
        similar_articles.sort(key=lambda x: float(x["credibility_score"].rstrip('%')) if x["credibility_score"] != "측정 불가" else 0, reverse=True)

        overall_credibility = sum(
            float(article["credibility_score"].rstrip('%')) for article in similar_articles if article["credibility_score"] != "측정 불가"
        ) / max(len([a for a in similar_articles if a["credibility_score"] != "측정 불가"]), 1)

        return jsonify({
            "original_sentiment": original_sentiment,
            "original_keywords": original_keywords,
            "credibility_score": f"{round(overall_credibility, 2)}%" if similar_articles else "측정 불가",
            "similar_articles": similar_articles[:4]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)