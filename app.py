from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

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

# API kiểm tra kết nối
@app.get('/')
async def root():
    return {"message": "Xin chao, Son day"}
