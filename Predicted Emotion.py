from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import sys
import io

# 표준 출력의 인코딩 설정을 UTF-8로 변경
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 감정 클래스 이름 매핑 (여기에 학습한 감정 이름을 넣으면 됩니다)
emotion_classes = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']  # 7개의 감정 클래스

# BERT 모델을 기반으로 하는 분류 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(emotion_classes))  # 'fc' -> 'classifier'로 수정

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 토큰의 출력을 사용
        x = self.dropout(pooled_output)
        x = self.classifier(x)  # 'fc' -> 'classifier'로 수정
        return x

# 모델 경로 설정
model_path = 'kobert_emotion_model.pth'  # 모델 파일 경로 (로컬에 저장된 파일)

# KoBERT 모델 로드 (사전 훈련된 KoBERT 모델 사용)
bert_model = BertModel.from_pretrained('monologg/kobert')
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')  # KoBERT용 토크나이저 사용

# BERTClassifier 모델 인스턴스 생성
model = BERTClassifier(bert_model).to(device)

# 저장된 모델 가중치 불러오기 (크기 불일치가 있을 경우 strict=False 사용)
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # strict=True로 불일치 키를 체크
    print("모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    sys.exit()

# 모델을 평가 모드로 설정
model.eval()

# 예측을 위한 함수 정의
def predict(text):
    # 입력 텍스트를 BERT 모델에 맞게 토크나이즈
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        # 모델에 입력을 주고 결과 얻기
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # 예측 결과에 softmax 적용하여 확률 계산
    prob = torch.nn.functional.softmax(outputs, dim=1)
    
    # 예측 클래스 추출 (확률이 가장 높은 클래스)
    predicted_class = torch.argmax(prob, dim=1).item()
    predicted_emotion = emotion_classes[predicted_class]  # 클래스 번호를 감정 이름으로 변환
    
    # 예측된 감정과 그 확률값 출력
    return predicted_emotion, prob[0].cpu().numpy()  # 확률값도 함께 반환

# 사용자로부터 텍스트 입력 받기
text = "오늘은 기분이 좋아요"

# 예측 실행
predicted_emotion, prob = predict(text)

# 출력 부분 수정 (감정과 확률)
print(f"Predicted Emotion: {predicted_emotion}")
print(f"Probabilities: {dict(zip(emotion_classes, prob))}")  # 각 감정 클래스별 확률값 출력
