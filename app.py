import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =============================
# 1. Đọc và xử lý dữ liệu
# =============================
du_lieu = pd.read_csv("data.csv")
du_lieu = du_lieu.drop(columns=["id", "Unnamed: 32"])
du_lieu["diagnosis"] = du_lieu["diagnosis"].map({"M": 0, "B": 1})

dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

X = du_lieu[dac_trung_quan_trong]
y = du_lieu["diagnosis"]

# =============================
# 2. Train mô hình Random Forest
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

mo_hinh = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)
mo_hinh.fit(X_train, y_train)

# =============================
# 3. Flask app
# =============================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", dac_trung=dac_trung_quan_trong)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # lấy dữ liệu JSON từ fetch
        gia_tri_nhap = [float(data[ten]) for ten in dac_trung_quan_trong]

        du_lieu_moi = pd.DataFrame([gia_tri_nhap], columns=dac_trung_quan_trong)
        du_doan_moi = mo_hinh.predict(du_lieu_moi)

        if du_doan_moi[0] == 0:
            ket_qua = "❌ Khối u ÁC TÍNH"
        else:
            ket_qua = "✅ Khối u LÀNH TÍNH"

        return jsonify({"result": ket_qua})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
