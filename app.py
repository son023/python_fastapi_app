from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
# Tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn, bạn có thể giới hạn để tăng bảo mật
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình
mon1_model = tf.keras.models.load_model('mon1_rnn.keras')
mon2_model = tf.keras.models.load_model('mon2_rnn.keras')
model= tf.keras.models.load_model("nhadat.keras")

model_5 = load_model('bilstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
encoder = joblib.load('encoder.pkl')
max_len = 100
class InputText(BaseModel):
    text: str


# Hàm hỗ trợ
def check(x):
    if x >= 9.0: return 'A+'
    elif x >= 8.5: return 'A'
    elif x >= 8: return 'B+'
    elif x >= 7: return 'B'
    elif x >= 6.5: return 'C+'
    elif x >= 6: return 'C'
    elif x >= 5: return 'D+'
    elif x >= 4: return 'D'
    else: return 'F'

def tb1(a, b, c, d, x):
    if x == 1:
        return (a * 10 + b * 10 + c * 20 + d * 60) / 100
    else:
        return (a * 10 + b * 20 + c * 20 + d * 50) / 100
    
class HouseFeatures(BaseModel):
    areaM2: float
    bedroom: int
    direction: int
    frontage: float
    lat: float
    legal: int
    long: float
    toiletCount: int

# Khai báo kiểu dữ liệu đầu vào
class InputData(BaseModel):
    input: list[float]

# API dự đoán với mô hình RNN
@app.post('/mon1')
async def predict_rnn(data: InputData):
    try:
        input_data = data.input
        a = input_data
        if len(input_data) != 3:
            raise HTTPException(status_code=400, detail="Dữ liệu đầu vào phải có 3 điểm")

        input_data = np.array(input_data).reshape(1, 1, 3)
        prediction = mon1_model.predict(input_data)
        ck = round(float(prediction[0][0]), 1)  # Chuyển đổi sang float
        tb = tb1(a[0], a[1], a[2], ck, 1)

        return {'prediction': ck, 'diem': tb, 'diemChu': check(tb)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# API dự đoán với mô hình LSTM
@app.post('/mon2')
async def predict_lstm(data: InputData):
    try:
        input_data = data.input
        a = input_data
        if len(input_data) != 3:
            raise HTTPException(status_code=400, detail="Dữ liệu đầu vào phải có 3 điểm")

        input_data = np.array(input_data).reshape(1, 1, 3)
        prediction = mon2_model.predict(input_data)
        ck = round(float(prediction[0][0]), 1)  # Chuyển đổi sang float
        tb = tb1(a[0], a[1], a[2], ck, 2)

        return {'prediction': ck, 'diem': tb, 'diemChu': check(tb)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Chuyển dữ liệu đầu vào thành định dạng numpy array
    input_data = np.array([[features.areaM2, features.bedroom, features.direction,
                            features.frontage, features.lat, features.legal,
                            features.long, features.toiletCount]])
    
    input_data = input_data.reshape((input_data.shape[1],1,1))
    prediction = model.predict(input_data)
    
    # Trả về kết quả dự đoán
    return {"predicted_price": float(prediction[0][0])}


@app.post("/predict_b5")
async def predict(input: InputText):
    input_text = input.text
    input_sequence = tokenizer.texts_to_sequences([input_text])  
    input_padded = pad_sequences(input_sequence, maxlen=max_len)  

    predicted_prob = model_5.predict(input_padded)
    predicted_label = (predicted_prob > 0.5).astype(int).flatten()

    decoded_label = encoder.inverse_transform(predicted_label)

    predicted_prob_value = float(predicted_prob[0][0])
    predicted_label_value = int(predicted_label[0])
    decoded_label_value = str(decoded_label[0])

    return {
    "input_text": input_text,
    "predicted_probability": round(predicted_prob_value, 4),
    "predicted_label": predicted_label_value,
    "decoded_label": decoded_label_value
    }

# API kiểm tra kết nối
@app.get('/')
async def root():
    return {"message": "Xin chao, Son day"}
