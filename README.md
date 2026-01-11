<img width="1558" height="673" alt="image" src="https://github.com/user-attachments/assets/ddddd94e-2190-4271-924e-397f726acf90" />


# 1. Giới thiệu đồ án môn học
- **Tên môn học**: Phân tích dữ liệu kinh doanh
- **Mã lớp**: IS403.P23
- **Tên đồ án**: A Comparative Analysis of Machine Learning, Recurrent Neural Networks and Deep Learning Models for Bitcoin Price Forecasting

# 2. Nhóm thực hiện
**Tên nhóm: Coconerd** 🥥

| Họ và tên          | MSSV     | Vai trò     | Liên hệ                     |
|--------------------|----------|-------------|-----------------------------|
|🌱  Trần Vũ Bão   | 22520124 | Team member | tranvubao2004@gmail.com          |
|🌱  Phan Thành Công       | 22520170 | Team member | ptcong.data@gmail.com          |
|🌱  Phan Thị Thủy Hiền | 22520423 | Team member | thuyhienphanthi2004@gmail.com |
|🌱  Nguyễn Đỗ Đức Minh | 22520872 | Team member   | nddminh2021@gmail.com          |

# 3. Nội dung đồ án
- Tìm hiểu và thực nghiệm về các mô hình học máy, học sâu trong phân tích và dự báo dữ liệu chuỗi thời gian về giá Bitcoin.
  - ARIMA
  - ARIMAX
  - XGBoost
  - LSTM
  - GRU
  - Transformer
- Dataset thực nghiệm được crawl về từ API của Yahoo Finance.

# 4. Phân tích khám phá dữ liệu (EDA)
<img width="876" height="547" alt="image" src="https://github.com/user-attachments/assets/aa02c862-a122-4e98-bed6-0365da62394c" />

<img width="878" height="295" alt="image" src="https://github.com/user-attachments/assets/e01e4d57-f811-4b05-8ff6-d66e72220e28" />


# 5. Kết quả thực nghiệm
## 5.1 Đánh giá kết quả của các mô hình trên tập test

🔎 **Bảng tổng hợp kết quả đánh giá**
<img width="784" height="170" alt="image" src="https://github.com/user-attachments/assets/a78ea7b1-b5d2-43ad-ba20-ae365b34696b" />

📈 **Trực quan hoá kết quả dự báo của các mô hình trên tập test**
  - ARIMA
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/cbc20e02-970a-46f5-a41d-bcb0b9dca398" />

  - ARIMAX
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/f1b9e06c-f392-4008-a0d4-b7d4c46abbbd" />

  - XGBoost
    <img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/b37da41f-fc1a-4246-b8c4-8ccfd7400ea6" />

  - LSTM
    <img width="1155" height="690" alt="image" src="https://github.com/user-attachments/assets/62adca86-31e9-4761-a499-091a3941f140" />

  - GRU
    <img width="1720" height="677" alt="image" src="https://github.com/user-attachments/assets/10741adb-d91c-4df1-b148-702c9a9cb3ce" />

  - Transformer
    <img width="1715" height="491" alt="image" src="https://github.com/user-attachments/assets/02c0a748-c4a6-468d-b117-92f0f6401a0c" />
    
📝 **Nhận xét**
- Kết quả cho thấy sự khác biệt đáng kể trong hiệu suất của các mô hình. Mô hình LSTM và GRU đạt được các chỉ số lỗi thấp nhất và giá trị R² cao nhất, thể hiện độ chính xác dự đoán vượt trội. Ngược lại, ARIMA và ARIMAX có sai số cao nhất và giá trị R² âm, cho thấy độ phù hợp kém. XGBoost và Transformer nằm ở mức trung bình, trong đó XGBoost hoạt động kém, còn Transformer thể hiện khả năng dự đoán vừa phải.

- ARIMA và ARIMAX là hai mô hình yếu nhất, với RMSE trên 37.000, MAE trên 31.900 và MAPE khoảng 41,7%. Giá trị R² âm (-2,845 và -2,86) cho thấy chúng không thể giải thích phương sai của giá Bitcoin, thậm chí còn kém hơn một mô hình đơn giản sử dụng trung bình. Dựa vào biểu đồ dự đoán trên tập test của ARIMA và ARIMAX cũng xác nhận điều này, khi các đường dự đoán gần như phẳng ở mức 40.000 và 30.000, trong khi dữ liệu thực tăng mạnh lên 90.000 vào cuối năm 2024. Điều này cho thấy mô hình tuyến tính truyền thống khó có thể mô phỏng được xu hướng phi tuyến, biến động cao của giá Bitcoin.

- XGBoost thể hiện cải thiện nhẹ với RMSE = 21.477, MAE = 15.641 và MAPE = 17,65%, nhưng R² vẫn âm (-0,97), cho thấy độ phù hợp thấp. Biểu đồ dự đoán trên tập test của XGBoost thể hiện dự đoán gần như phẳng quanh mức 80.000, không phản ánh đúng biến động thực từ 70.000 đến 80.000. Dù có khả năng học phi tuyến, XGBoost vẫn gặp khó khăn trong việc nắm bắt tính chuỗi thời gian.

- Ngược lại, LSTM và GRU vượt trội nhất, với RMSE lần lượt là 2.119 và 2.808, MAE là 1.558 và 2.085, MAPE lần lượt là 2,25% và 2,85%. Giá trị R² cao (0,9867 và 0,978) cho thấy hơn 97% phương sai trong dữ liệu test được giải thích. Biểu đồ dự đoán trên tập test của LSTM và GRU cho thấy kết quả dự đoán sát với dữ liệu thực tế, đặc biệt LSTM theo sát đỉnh 60.000 và 90.000, trong khi GRU phản ánh tốt xu hướng giảm và phục hồi giai đoạn đầu 2025.

- Mô hình Transformer đạt kết quả trung bình với RMSE = 15.798, MAE = 11.017 và MAPE = 13,16%. R² chỉ đạt 0,25, tức chỉ giải thích được 25% phương sai. Biểu đồ dự đoán trên tập test của Transformer cho thấy mô hình theo xu hướng chung nhưng đánh giá thấp đỉnh 60.000 vào năm 2024, chỉ dự đoán khoảng 40.000. Điều này phản ánh khả năng học có giới hạn với các biến động cực đoan, và mô hình có thể cần được tối ưu thêm.

## 5.2 Trực quan hoá kết quả dự báo giá bitcoin trong 30 ngày tiếp theo
  - ARIMA
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/fdf278a7-d4ca-4ce6-afeb-92b55bb713d7" />

  - ARIMAX
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/1c5b49ca-6471-40fc-94c1-e04c5970c0a0" />
    
  - XGBoost
    <img width="986" height="351" alt="image" src="https://github.com/user-attachments/assets/f38de120-4f30-4d5b-8a28-4036bfc3bdd0" />

  - LSTM
    <img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/1a3d65e9-e177-4b6d-ac34-e242b729d613" />

  - GRU
    <img width="1621" height="690" alt="image" src="https://github.com/user-attachments/assets/27d73886-3bc6-48be-99b8-a0b61ae127d5" />

  - Transformer
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/475ac72e-9d99-4a3c-9db8-8c9f8bc3a179" />

## 5.3 Kết luận
Dự án so sánh các mô hình dự báo giá Bitcoin ngắn hạn bao gồm ARIMA, ARIMAX, XGBoost, LSTM, GRU và Transformer. Kết quả cho thấy:

  🔴 ARIMA/ARIMAX: Sai số rất lớn, R² âm → Dự báo kém, không phù hợp với dữ liệu biến động cao như giá Bitcoin.
  
  🟠 XGBoost: Cải thiện nhẹ nhưng vẫn không nắm bắt được tính chuỗi → Dự báo gần như phẳng, R² âm.
  
  🟡 Transformer: Dự đoán tốt hơn, nắm bắt được xu hướng nhưng đánh giá thấp các đỉnh lớn → Cần tối ưu thêm.
  
  🟢 LSTM/GRU: Hiệu quả nhất với sai số thấp, R² trên 0.97 → Dự đoán sát thực tế, phù hợp cho dữ liệu chuỗi phức tạp.
  
  
👉 Các mô hình deep learning, đặc biệt là LSTM và GRU, là lựa chọn phù hợp để mô phỏng tính phi tuyến và biến động mạnh của thị trường tiền mã hoá.
