"""Ba mô hình phân lớp cổ điển trên đặc trưng TF-IDF.

M1 Naive Bayes theo hướng sinh, M2 Logistic Regression theo hướng phân biệt xác
suất, M3 Linear SVM theo hướng tối đa hóa lề. Dùng cả ba để phép so sánh với
PhoBERT không bị dễ dãi.
"""

import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import cau_hinh as ch
from . import danh_gia as dg


def danh_sach_mo_hinh():
    # M1 không đặt class_weight vì xác suất tiên nghiệm của Naive Bayes vốn đã
    # phản ánh phân bố lớp, can thiệp sẽ phá vỡ ý nghĩa xác suất của mô hình
    return [
        ("M1_NaiveBayes", MultinomialNB(alpha=0.1)),
        ("M2_LogisticRegression", LogisticRegression(
            max_iter=1000, C=10.0, class_weight="balanced")),
        ("M3_LinearSVM", LinearSVC(C=1.0, class_weight="balanced")),
    ]


def huan_luyen_va_danh_gia(X_train, y_train, X_test, y_test):
    """Huấn luyện cả ba mô hình rồi đo trên tập kiểm thử."""
    ket_qua = {}
    du_doan_luu = {}
    mo_hinh_luu = {}

    for ten, mo_hinh in danh_sach_mo_hinh():
        t0 = time.time()
        mo_hinh.fit(X_train, y_train)
        giay = time.time() - t0
        y_doan = mo_hinh.predict(X_test)

        ket_qua[ten] = dg.do_ket_qua(ten, y_test, y_doan, giay)
        dg.luu_ket_qua(ten, ket_qua[ten])
        du_doan_luu[ten] = y_doan
        mo_hinh_luu[ten] = mo_hinh

    return ket_qua, du_doan_luu, mo_hinh_luu


def kiem_dinh_cheo(tr):
    """Kiểm định chéo 5 lớp phân tầng, chạy trên TẬP HUẤN LUYỆN.

    Mục đích là biết kết quả dao động bao nhiêu khi đổi tập huấn luyện, chứ
    không phải làm accuracy chính xác hơn.

    Vectorizer phải được fit lại ở TỪNG lớp gấp, nếu fit một lần bên ngoài vòng
    lặp thì mỗi lớp gấp kiểm thử đều bị rò rỉ. Gói vào Pipeline để thư viện tự
    quản lý thứ tự.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline

    from .dac_trung import tao_vectorizer

    skf = StratifiedKFold(n_splits=ch.SO_LOP_GAP, shuffle=True,
                          random_state=ch.HAT_GIONG)
    # tolist() vì ArrowStringArray của pandas 3.x không hỗ trợ cách lấy phần tử
    # bằng mảng chỉ số mà cross_validate dùng để chia lớp gấp
    X = tr["van_ban_tach_tu"].tolist()
    y = tr["chu_de"].tolist()

    ket_qua_cv = {}
    for ten, mo_hinh in danh_sach_mo_hinh():
        duong_ong = Pipeline([("tfidf", tao_vectorizer()), ("mo_hinh", mo_hinh)])
        t0 = time.time()
        diem = cross_validate(duong_ong, X, y, cv=skf,
                              scoring=["accuracy", "f1_macro"], n_jobs=1)
        a = diem["test_accuracy"]
        f = diem["test_f1_macro"]
        ket_qua_cv[ten] = {
            "acc_tb": round(float(a.mean()), 4),
            "acc_lech": round(float(a.std(ddof=1)), 4),
            "f1_tb": round(float(f.mean()), 4),
            "f1_lech": round(float(f.std(ddof=1)), 4),
        }
        dg.luu_ket_qua(ten, ket_qua_cv[ten], khoa="ket_qua_cv")
        print(ten, "(", round(time.time() - t0, 1), "giây )")
        print("   accuracy trung bình:", ket_qua_cv[ten]["acc_tb"],
              "độ lệch chuẩn", ket_qua_cv[ten]["acc_lech"])
        print("   macro-F1 trung bình:", ket_qua_cv[ten]["f1_tb"],
              "độ lệch chuẩn", ket_qua_cv[ten]["f1_lech"])
        print()
    return ket_qua_cv
