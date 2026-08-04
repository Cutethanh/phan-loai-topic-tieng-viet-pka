"""Điểm khởi chạy của toàn bộ dự án.

Cách dùng:

    python chay.py --buoc chuan_bi     Tải dữ liệu, khử trùng lặp, tách từ
    python chay.py --buoc thong_ke     In bảng thống kê mô tả và vẽ biểu đồ
    python chay.py --buoc co_dien      Huấn luyện và đánh giá ba mô hình cổ điển
    python chay.py --buoc kiem_dinh    Kiểm định chéo 5 lớp
    python chay.py --buoc phobert      Tinh chỉnh PhoBERT, cần GPU
    python chay.py --buoc du_doan      Thử dự đoán trên vài văn bản mẫu
    python chay.py --buoc tong_hop     In bảng tổng hợp kết quả
    python chay.py --buoc tat_ca       Chạy toàn bộ nhánh cổ điển từ đầu tới cuối

Tùy chọn thêm:

    --mau_thu              Chỉ dùng 300 văn bản mỗi lớp, để chạy thử cho nhanh
    --cat head_tail        Đổi cách cắt văn bản cho PhoBERT
    --lam_lai              Bỏ qua tệp đã tách từ và làm lại từ đầu
"""

import argparse
import os
import sys

from src import cau_hinh as ch
from src import danh_gia as dg
from src import dac_trung, du_doan, mo_hinh_co_dien, tien_xu_ly


def buoc_chuan_bi(a):
    df = tien_xu_ly.chuan_bi_du_lieu(mau_thu=a.mau_thu, bat_buoc_lam_lai=a.lam_lai)
    tr, te, nhan_lop = tien_xu_ly.chia_tap(df)
    print()
    print("Tập huấn luyện:", len(tr), "văn bản")
    print("Tập kiểm thử  :", len(te), "văn bản")
    print("Số lớp        :", len(nhan_lop))
    return df


def buoc_thong_ke(a):
    df = tien_xu_ly.chuan_bi_du_lieu(mau_thu=a.mau_thu, bat_buoc_lam_lai=a.lam_lai)
    bang = dg.ve_phan_bo_nhan(df)
    print()
    print("BẢNG THỐNG KÊ MÔ TẢ THEO CHỦ ĐỀ")
    b = bang.copy()
    b.index = [ch.TEN_HIEN_THI.get(i, i) for i in b.index]
    b.loc["TỔNG"] = b.sum()
    print(b.to_string())
    print()

    import pandas as pd
    print("BẢNG PHÂN BỐ THEO TÒA SOẠN (kiểm tra chéo rò rỉ dữ liệu)")
    bb = pd.crosstab(df["chu_de"], df["toa_soan"])
    bb.index = [ch.TEN_HIEN_THI.get(i, i) for i in bb.index]
    print(bb.to_string())
    print()
    print("Mỗi chủ đề đều có bài từ bao nhiêu nguồn báo:",
          (bb > 0).sum(axis=1).unique(), "nguồn")
    print()

    print("ĐỘ DÀI VĂN BẢN (đếm theo khoảng trắng, chưa tách từ)")
    print(df.groupby("tap")["so_tu"].agg(
        trung_binh="mean", trung_vi="median",
        nho_nhat="min", lon_nhat="max").round(1).to_string())
    print()

    tong = bang["tong"]
    print("Mức lệch giữa lớp lớn nhất và lớp nhỏ nhất:",
          round(tong.max() / tong.min(), 2), "lần")
    print("Vì lệch như vậy nên chỉ số chính của báo cáo là macro-F1, không phải accuracy.")


def buoc_co_dien(a):
    df = tien_xu_ly.chuan_bi_du_lieu(mau_thu=a.mau_thu, bat_buoc_lam_lai=a.lam_lai)
    tr, te, nhan_lop = tien_xu_ly.chia_tap(df)

    print()
    dac_trung.minh_hoa_ro_ri()
    print()

    X_train, X_test, vec = dac_trung.xay_dung_dac_trung(tr, te)
    # .tolist() thay vi .values: tu pandas 3.x, cot chuoi duoc luu bang
    # ArrowStringArray thay vi mang numpy. Kieu nay khong ho tro day du
    # cach lay phan tu bang mang chi so ma scikit-learn dung ben trong,
    # nen chuyen ve list Python thuan cho chac chan.
    y_train, y_test = tr["chu_de"].tolist(), te["chu_de"].tolist()
    print()

    kq, du_doan_luu, mo_hinh_luu = mo_hinh_co_dien.huan_luyen_va_danh_gia(
        X_train, y_train, X_test, y_test)

    tot = max(kq, key=lambda k: kq[k]["macro_f1"])
    print("Mô hình cổ điển có macro-F1 cao nhất:", tot)
    print()
    dg.bao_cao_tung_lop(y_test, du_doan_luu[tot], nhan_lop)
    dg.ve_ma_tran_nham_lan(y_test, du_doan_luu[tot], nhan_lop, tot)

    # Lưu Logistic Regression để triển khai, vì mô hình này cho xác suất thật sự.
    # LinearSVC chỉ cho khoảng cách tới mặt phân chia, không phải xác suất.
    du_doan.luu_mo_hinh(vec, mo_hinh_luu["M2_LogisticRegression"])
    return y_test, du_doan_luu, nhan_lop


def buoc_kiem_dinh(a):
    df = tien_xu_ly.chuan_bi_du_lieu(mau_thu=a.mau_thu, bat_buoc_lam_lai=a.lam_lai)
    tr, _, _ = tien_xu_ly.chia_tap(df)
    print()
    print("Kiểm định chéo", ch.SO_LOP_GAP, "lớp trên tập huấn luyện")
    print("Phần này fit lại TF-IDF nhiều lần nên chạy lâu hơn bước co_dien.")
    print()
    mo_hinh_co_dien.kiem_dinh_cheo(tr)


def buoc_phobert(a):
    from src import mo_hinh_phobert
    print("LƯU Ý: mô đun PhoBERT chưa được chạy thử, xem chú thích đầu tệp")
    print("       src/mo_hinh_phobert.py")
    print()
    if not mo_hinh_phobert.kiem_tra_gpu():
        print()
        print("Môi trường GPU chưa sẵn sàng. Dừng ở đây.")
        return
    print()
    df = tien_xu_ly.chuan_bi_du_lieu(mau_thu=a.mau_thu, bat_buoc_lam_lai=a.lam_lai)
    tr, te, nhan_lop = tien_xu_ly.chia_tap(df)
    mo_hinh_phobert.tinh_chinh(tr, te, nhan_lop, kieu_cat=a.cat)


def buoc_du_doan(a):
    du_doan.thu_du_doan()


def buoc_tong_hop(a):
    dg.in_bang_tong_hop()


def main():
    p = argparse.ArgumentParser(
        description="Phân loại chủ đề bài viết tiếng Việt trên kho ngữ liệu VNTC")
    p.add_argument("--buoc", required=True,
                   choices=["chuan_bi", "thong_ke", "co_dien", "kiem_dinh",
                            "phobert", "du_doan", "tong_hop", "tat_ca"])
    p.add_argument("--mau_thu", action="store_true",
                   help="Chỉ dùng 300 văn bản mỗi lớp, để chạy thử cho nhanh")
    p.add_argument("--cat", default=None, choices=["head", "head_tail"],
                   help="Cách cắt văn bản cho PhoBERT")
    p.add_argument("--lam_lai", action="store_true",
                   help="Bỏ qua tệp đã tách từ và làm lại từ đầu")
    a = p.parse_args()

    ch.bao_dam_thu_muc()
    if a.mau_thu:
        ch.THU_MUC_KET_QUA = os.path.join(ch.THU_MUC_KET_QUA, "mau_thu")
        ch.TEP_THONG_KE = os.path.join(ch.THU_MUC_KET_QUA, "thong_ke_du_lieu.json")
        ch.TEP_KET_QUA = os.path.join(ch.THU_MUC_KET_QUA, "ket_qua.json")
        ch.TEP_MO_HINH = os.path.join(ch.THU_MUC_MO_HINH,
                                      "mo_hinh_phan_loai_mau_thu.joblib")
        os.makedirs(ch.THU_MUC_KET_QUA, exist_ok=True)
        print("CHẾ ĐỘ CHẠY THỬ: mọi kết quả ghi vào", ch.THU_MUC_KET_QUA)
        print()

    if a.buoc == "tat_ca":
        print("### BƯỚC 1: CHUẨN BỊ DỮ LIỆU ###")
        buoc_chuan_bi(a)
        print()
        print("### BƯỚC 2: THỐNG KÊ MÔ TẢ ###")
        buoc_thong_ke(a)
        print()
        print("### BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ ###")
        y_test, du_doan_luu, nhan_lop = buoc_co_dien(a)
        print()
        print("### BƯỚC 4: KIỂM ĐỊNH CHÉO ###")
        buoc_kiem_dinh(a)
        print()
        print("### BƯỚC 5: THỬ DỰ ĐOÁN ###")
        buoc_du_doan(a)
        print()
        print("### BƯỚC 6: BẢNG TỔNG HỢP ###")
        dg.in_bang_tong_hop(nhan_lop, y_test, du_doan_luu)
        return

    ham = {"chuan_bi": buoc_chuan_bi, "thong_ke": buoc_thong_ke,
           "co_dien": buoc_co_dien, "kiem_dinh": buoc_kiem_dinh,
           "phobert": buoc_phobert, "du_doan": buoc_du_doan,
           "tong_hop": buoc_tong_hop}[a.buoc]
    ham(a)


if __name__ == "__main__":
    sys.exit(main())
