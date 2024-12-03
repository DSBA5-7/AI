from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import re
import os
import jpype
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from konlpy.tag import Okt

# Java 환경변수 설정
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home'

# JVM 초기화
jvm_path = "/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/lib/server/libjvm.dylib"
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, "-Dfile.encoding=UTF8", convertStrings=False)

# Flask 초기화
app = Flask(__name__)
CORS(app)

# 모델 초기화
kw_model = KeyBERT()
okt = Okt()

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

# 크롤링 함수: 여러 HTML 구조 처리
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
def extract_keywords_with_pos(text):
    clean_text = preprocess_text(text)

    # KeyBERT 키워드 추출
    keybert_keywords = kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1, 1), top_n=10)
    keybert_keywords = [kw[0] for kw in keybert_keywords]

    # POS tagging으로 주어, 지역, 날짜 추출
    subjects, locations, dates = extract_contextual_keywords(clean_text)

    # 키워드 통합
    combined_keywords = list(set(keybert_keywords + subjects + locations + dates))
    return combined_keywords[:10]  # 최대 10개의 키워드 반환

# 외부 기사 검색 함수
def search_similar_articles(keywords):
    # Google News API 또는 다른 API 호출
    pass

# 기사 비교 및 신뢰도 평가 함수
def evaluate_trustworthiness(original_article, similar_articles):
    # 키워드 유사도, 감정 일치도 계산
    pass

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
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), top_n=10)
        extracted_keywords = [kw[0] for kw in keywords]

        # 유사 기사 검색
        similar_articles = search_similar_articles(extracted_keywords)

        # 각 유사 기사에서 텍스트 크롤링
        similar_articles_texts = []
        for article in similar_articles:
            try:
                article_text = crawl_text_from_url(article['url'])
                similar_articles_texts.append({
                    "title": article['title'],
                    "url": article['url'],
                    "text": article_text
                })
            except Exception as e:
                # 특정 기사 크롤링 실패 시 스킵
                print(f"Failed to crawl article at {article['url']}: {str(e)}")

        # 유사도 계산
        candidate_texts = [article["text"] for article in similar_articles_texts]
        similarities = calculate_similarity(text, candidate_texts)

        # 감정 분석
        original_emotion = analyze_emotion(text)
        similar_articles_emotions = [analyze_emotion(article["text"]) for article in similar_articles_texts]

        # 결과 구성
        results = []
        for article, similarity, emotion in zip(similar_articles_texts, similarities, similar_articles_emotions):
            results.append({
                "title": article['title'],
                "url": article['url'],
                "similarity": similarity,
                "emotion": emotion
            })

        return jsonify({
            "original_keywords": extracted_keywords,
            "original_emotion": original_emotion,
            "similar_articles": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500