import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse_func, mean_absolute_error as mae_func
import numpy as np

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('study_score_prediction_data.csv')

# 2. Chọn các cột cần thiết
X = df[['hours_studied', 'attendance', 'previous_score']]  # Các biến độc lập
y = df['current_score']  # Biến phụ thuộc

# 3. Chuẩn hóa dữ liệu để có trung bình bằng 0 và độ lệch chuẩn bằng 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# 6. Huấn luyện mô hình
model.fit(X_train, y_train)

# 7. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 8. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Tính RMSE bằng căn bậc hai của MSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Tính Adjusted R-squared
n = X_test.shape[0]  # Số mẫu trong tập kiểm tra
p = X_test.shape[1]  # Số biến độc lập
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 9. Dự báo điểm số của sinh viên
print(f"MSE: {mse:.2f}")            # Sai số bình phương trung bình
print(f"RMSE: {rmse:.2f}")          # Căn bậc hai của MSE
print(f"MAE: {mae:.2f}")            # Sai số trung bình tuyệt đối
print(f"R²: {r2:.2f}")              # Hệ số xác định
print(f"Adjusted R²: {adjusted_r2:.2f}")  # Hệ số xác định điều chỉnh
print(f"Dự đoán điểm số: {y_pred}")

# Phân loại học sinh dựa trên điểm số dự đoán
for score in y_pred:
    if score >= 70:
        print(f"Điểm số {score:.2f}: Học sinh học tốt.")
    else:
        print(f"Điểm số {score:.2f}: Học sinh học kém.")