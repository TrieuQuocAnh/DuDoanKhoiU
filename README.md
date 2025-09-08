📌 Bài toán

Ứng dụng dự đoán khối u vú lành tính hay ác tính dựa trên các chỉ số xét nghiệm mô học.

Dữ liệu sử dụng: bộ dữ liệu Breast Cancer Wisconsin (Diagnostic) thường được dùng trong học máy.

Đầu vào: các đặc trưng về hình dạng tế bào (ví dụ: số điểm lõm, bán kính, diện tích...).

Đầu ra: dự đoán khối u lành tính (B) hay ác tính (M).

⚙️ Công nghệ sử dụng

Python làm ngôn ngữ chính.

Pandas để xử lý dữ liệu (đọc CSV, làm sạch dữ liệu).

Scikit-learn để train mô hình học máy.

Flask để xây dựng ứng dụng web backend.

HTML + CSS + JavaScript để xây dựng giao diện web.

(Có thể mở rộng với Bootstrap / TailwindCSS nếu muốn giao diện chuyên nghiệp hơn).

🧠 Thuật toán

Sử dụng Random Forest Classifier (thuật toán học máy dạng Ensemble Learning).

Nguyên lý: tạo ra nhiều cây quyết định (Decision Trees) trên các tập dữ liệu con và đặc trưng con, sau đó lấy bỏ phiếu đa số để dự đoán.

Ưu điểm:

Chính xác cao, chống overfitting.

Tự động chọn đặc trưng quan trọng.

Trong bài toán này: chọn 5 đặc trưng quan trọng nhất để đưa vào mô hình:

concave points_mean (trung bình số điểm lõm)

concave points_worst (số điểm lõm nhiều nhất)

area_worst (diện tích lớn nhất)

concavity_mean (trung bình độ lõm)

radius_worst (bán kính lớn nhất)

<img width="644" height="853" alt="image" src="https://github.com/user-attachments/assets/5573c30f-4a99-45e0-b3c6-7ed0bed7b6ca" />

<img width="606" height="819" alt="image" src="https://github.com/user-attachments/assets/495073cc-cef5-4220-8058-a77796835ea2" />


