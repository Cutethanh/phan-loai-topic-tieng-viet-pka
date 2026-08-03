"""Trích xuất đặc trưng bằng TF-IDF
"""

import math
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from . import cau_hinh as ch


def tao_vectorizer():
    return TfidfVectorizer(**ch.THAM_SO_TFIDF)


def xay_dung_dac_trung(tr, te):
    """Trả về ma trận đặc trưng của hai tập cùng vectorizer đã fit.

    Thứ tự hai bước dưới đây là bắt buộc và không được đảo.
    """
    vec = tao_vectorizer()
    t0 = time.time()

    # tolist() vì từ pandas 3.x cột chuỗi dùng ArrowStringArray, không tương
    # thích hoàn toàn với cách scikit-learn lấy phần tử bằng mảng chỉ số
    X_train = vec.fit_transform(tr["van_ban_tach_tu"].tolist())
    X_test = vec.transform(te["van_ban_tach_tu"].tolist())

    print("TF-IDF xong sau", round(time.time() - t0, 1), "giây")
    print("   Kích thước ma trận train:", X_train.shape)
    print("   Kích thước ma trận test :", X_test.shape)
    print("   Số đặc trưng học được từ tập train:", len(vec.vocabulary_))
    return X_train, X_test, vec


def minh_hoa_ro_ri():
    """In ví dụ số cho thấy chênh lệch giữa cách làm đúng và cách làm sai.

    Chỉ dùng để minh họa trong báo cáo, không tham gia quá trình huấn luyện.
    """
    def idf(N, df_t):
        return math.log((1 + N) / (1 + df_t)) + 1

    print("MINH HỌA LỖI RÒ RỈ DỮ LIỆU")
    print("Giả sử 5 văn bản, 4 để huấn luyện và 1 để kiểm thử.")
    print()
    print("Trường hợp 1: từ có mặt ở cả hai tập")
    print("   Làm đúng (N = 4, df = 1): idf =", round(idf(4, 1), 4))
    print("   Làm sai  (N = 5, df = 2): idf =", round(idf(5, 2), 4))
    print("   Chênh lệch:", round(idf(4, 1) - idf(5, 2), 4))
    print()
    print("Trường hợp 2: từ chỉ xuất hiện trong tập kiểm thử")
    print("   Làm đúng: từ không nằm trong bộ từ vựng nên bị bỏ qua, đúng như")
    print("             điều sẽ xảy ra khi hệ thống chạy thật.")
    print("   Làm sai : từ có cột riêng với idf =", round(idf(5, 1), 4))
    print("             tức mô hình khai thác một tín hiệu không được phép biết.")
