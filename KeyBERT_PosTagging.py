from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Flask 초기화
app = Flask(__name__)
CORS(app)

# 모델 초기화
kw_model = KeyBERT()

# 감정 분석 모델 로드
emotion_model_path = "/Users/pjy/Desktop/DSBA5-7/DSBAGit/AI/kobert_emotion_model.pth"
emotion_model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)
emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
emotion_model.eval()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

# 크롤링 함수
def crawl_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 제목 추출
        title_candidates = ['h1', 'h2', 'title']
        title_text = ""
        for candidate in title_candidates:
            title = soup.find(candidate)
            if title:
                title_text = title.get_text(strip=True)
                break

        # 본문 추출
        body_candidates = [
            {'id': 'content'},
            {'class': 'article-body'},
            {'class': 'content-body'},
            {'class': 'post-content'},
            {'class': 'entry-content'},
            {'class': 'news-body'}
        ]
        body_text = ""
        for candidate in body_candidates:
            body = soup.find('div', candidate)
            if body:
                body_text = body.get_text(strip=True)
                break

        return f"{title_text} {body_text}"

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch the URL: {str(e)}")

# 감정 분석 함수
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    emotion_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return emotion_map.get(predicted_class, "Unknown")

# 키워드 추출 함수
def extract_keywords_with_regex(text):
    clean_text = preprocess_text(text)

    # KeyBERT 키워드 추출
    keybert_keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1, 1), top_n=10)
    keybert_keywords = [kw[0] for kw in keybert_keywords]

    return keybert_keywords[:10]  # 최대 10개의 키워드 반환

# 유사 기사 검색 함수
def search_similar_articles(keywords):
    # 예제: Google News API 또는 다른 API 호출
    # Placeholder로 9개의 유사 기사 반환
    return [{"title": f"Sample Article {i}", "url": f"http://example.com/article{i}"} for i in range(1, 10)]

# 신뢰도 평가 함수
def calculate_trustworthiness(input_text, similar_articles):
    weights = {"keyword": 0.4, "emotion": 0.3, "topic": 0.2, "sensationalism": 0.1}

    # 입력된 기사 키워드 및 감정 분석
    input_keywords = extract_keywords_with_regex(input_text)
    input_emotion = analyze_emotion(input_text)

    total_keyword_similarity = 0
    total_emotion_similarity = 0
    sensationalism_score = 1

    for article in similar_articles:
        article_keywords = article["keywords"]
        keyword_similarity = len(set(input_keywords) & set(article_keywords)) / len(set(input_keywords) | set(article_keywords))
        total_keyword_similarity += keyword_similarity

        article_emotion = article["emotion"]
        if input_emotion == article_emotion:
            total_emotion_similarity += 1

        # 부정적 감정이 많은 경우 자극성 점수 감소
        if article_emotion == "Negative":
            sensationalism_score -= 0.1

    keyword_similarity_score = total_keyword_similarity / len(similar_articles)
    emotion_similarity_score = total_emotion_similarity / len(similar_articles)
    topic_similarity_score = 0.8  # Placeholder 값

    trustworthiness_score = (
        weights["keyword"] * keyword_similarity_score +
        weights["emotion"] * emotion_similarity_score +
        weights["topic"] * topic_similarity_score +
        weights["sensationalism"] * sensationalism_score
    )

    return {
        "trustworthiness_score": trustworthiness_score,
        "keyword_similarity": keyword_similarity_score,
        "emotion_similarity": emotion_similarity_score,
        "topic_similarity": topic_similarity_score,
        "sensationalism_score": sensationalism_score
    }

# Flask API 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url', '')

    if not url or not url.startswith("http"):
        return jsonify({"error": "Invalid or missing URL"}), 400

    try:
        # URL에서 텍스트 크롤링
        text = crawl_text_from_url(url)

        # 키워드 추출
        keywords = extract_keywords_with_regex(text)

        # 유사 기사 검색
        similar_articles = search_similar_articles(keywords)

        # 각 유사 기사에서 텍스트 크롤링 및 분석
        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                article["text"] = article_text
                article["keywords"] = extract_keywords_with_regex(article_text)
                article["emotion"] = analyze_emotion(article_text)
            except Exception as e:
                article["error"] = str(e)

        # 신뢰도 계산
        trustworthiness = calculate_trustworthiness(text, similar_articles)

        return jsonify({
            "original_keywords": keywords,
            "original_emotion": analyze_emotion(text),
            "similar_articles": similar_articles,
            "trustworthiness": trustworthiness
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask 실행
if __name__ == '__main__':
    app.run(debug=True)