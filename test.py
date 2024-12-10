from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import re
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import gdown
import os

# Flask 초기화
app = Flask(
    __name__, 
    template_folder="../FE/templates",  # HTML 폴더 경로
    static_folder="../FE/static"        # 정적 파일 경로 (CSS, JS 등)
)
CORS(app)

# KeyBERT 모델 초기화
kw_model = KeyBERT()

# Google Drive에서 .pth 파일 다운로드
def download_model_from_drive():
    file_id = "1uS2PvnVaX1geCbv34MoWi7y1TF9wM_I6"
    output_path = 'kobert_emotion_model.pth'
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    return output_path

# 감정 분석 모델 로드
emotion_model_path = download_model_from_drive()
emotion_model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=7)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
emotion_model.eval()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

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
    if not text.strip():
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
def calculate_credibility(emotion, similarity_score, sentiment):
    emotion_scores = {
        "공포": 30,
        "놀람": 50,
        "분노": 20,
        "슬픔": 25,
        "중립": 40,
        "행복": 60,
        "혐오": 10
    }
    sentiment_scores = {"양성": 40, "음성": 40, "중립": 20}

    emotion_score = emotion_scores.get(emotion, 0)
    sentiment_score = sentiment_scores.get(sentiment, 30)
    credibility = (
        EMOTION_WEIGHT * emotion_score
        + SIMILARITY_WEIGHT * similarity_score
        + SENTIMENT_WEIGHT * sentiment_score
    )
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
        original_emotion = analyze_emotion(text)
        original_sentiment = classify_sentiment(text, positive_words, negative_words)
        original_keywords = extract_keywords_as_single_phrase(text)

        similar_articles = search_similar_articles(original_keywords)
        if not similar_articles:
            return jsonify({
                "original_emotion": original_emotion,
                "original_sentiment": original_sentiment,
                "original_keywords": original_keywords,
                "credibility_score": "측정 불가",
                "similar_articles": []
            })

        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                similarity_score = calculate_similarity(text, article_text)
                credibility_score = calculate_credibility(original_emotion, similarity_score, original_sentiment)
                article["credibility_score"] = f"{credibility_score}%"
            except Exception:
                article["credibility_score"] = "측정 불가"

        # 신뢰도 높은 순으로 정렬
        similar_articles.sort(key=lambda x: float(x["credibility_score"].rstrip('%')) if x["credibility_score"] != "측정 불가" else 0, reverse=True)

        overall_credibility = sum(
            float(article["credibility_score"].rstrip('%')) for article in similar_articles if article["credibility_score"] != "측정 불가"
        ) / max(len([a for a in similar_articles if a["credibility_score"] != "측정 불가"]), 1)

        return jsonify({
            "original_emotion": original_emotion,
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
