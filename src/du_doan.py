"""Nạp mô hình đã huấn luyện và dự đoán chủ đề cho văn bản mới
"""

import os

import joblib
import numpy as np

from . import cau_hinh as ch
from .tien_xu_ly import tien_xu_ly


def luu_mo_hinh(vectorizer, mo_hinh, duong_dan=None):
    duong_dan = duong_dan or ch.TEP_MO_HINH
    joblib.dump({
        "vectorizer": vectorizer,
        "mo_hinh": mo_hinh,
        "nhan_lop": list(mo_hinh.classes_),
    }, duong_dan)
    print("Đã lưu mô hình vào", duong_dan)
    return duong_dan


class BoPhanLoai:
    """Bọc mô hình đã huấn luyện. Nạp một lần lúc khởi tạo, dùng nhiều lần."""

    def __init__(self, duong_dan=None):
        duong_dan = duong_dan or ch.TEP_MO_HINH
        if not os.path.exists(duong_dan):
            raise FileNotFoundError(
                "Không thấy tệp mô hình " + duong_dan +
                ". Hãy chạy huấn luyện trước: python chay.py --buoc co_dien")
        goi = joblib.load(duong_dan)
        self.vectorizer = goi["vectorizer"]
        self.mo_hinh = goi["mo_hinh"]
        self.nhan_lop = goi["nhan_lop"]

    def du_doan(self, van_ban, top_k=3):
        """Nhận văn bản thô, trả về danh sách nhãn kèm độ tin cậy."""
        x = self.vectorizer.transform([tien_xu_ly(van_ban)])
        if hasattr(self.mo_hinh, "predict_proba"):
            p = self.mo_hinh.predict_proba(x)[0]
        else:
            # LinearSVC chỉ cho khoảng cách tới mặt phân chia. Ép qua softmax ra
            # số trong khoảng 0 tới 1 nhưng không có ý nghĩa hiệu chuẩn, chỉ dùng
            # để xếp hạng.
            d = self.mo_hinh.decision_function(x)[0]
            e = np.exp(d - d.max())
            p = e / e.sum()
        thu_tu = np.argsort(p)[::-1][:top_k]
        return [{"chu_de": self.mo_hinh.classes_[i],
                 "ten_hien_thi": ch.TEN_HIEN_THI.get(self.mo_hinh.classes_[i],
                                                     self.mo_hinh.classes_[i]),
                 "do_tin_cay": round(float(p[i]), 4)} for i in thu_tu]

    def du_doan_nhieu(self, danh_sach, top_k=3):
        return [self.du_doan(vb, top_k) for vb in danh_sach]


VAN_BAN_THU = [
    "Đội tuyển Việt Nam giành chiến thắng 2-0 trong trận đấu tối qua tại sân Mỹ Đình, tiền đạo ghi bàn ở phút 78.",
    "Giá cổ phiếu ngân hàng đồng loạt tăng trần, chỉ số VN-Index vượt mốc 1.200 điểm trong phiên giao dịch hôm nay.",
    "Các nhà khoa học vừa công bố phát hiện mới về cấu trúc protein giúp hiểu rõ hơn cơ chế phân chia tế bào.",
    "Bộ Y tế khuyến cáo người dân tiêm nhắc lại vắc xin và rửa tay thường xuyên để phòng bệnh hô hấp mùa lạnh.",
    "Bộ vi xử lý thế hệ mới hỗ trợ bộ nhớ DDR5 và có hiệu năng đồ họa tích hợp cao hơn đời trước.",
]


def thu_du_doan(duong_dan=None):
    bpl = BoPhanLoai(duong_dan)
    for vb in VAN_BAN_THU:
        print("Văn bản:", vb[:70])
        for r in bpl.du_doan(vb):
            print("     ", r["ten_hien_thi"], "_ độ tin cậy", r["do_tin_cay"])
        print()
