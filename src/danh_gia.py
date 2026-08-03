"""Tính các chỉ số đánh giá và vẽ ma trận nhầm lẫn
"""

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from . import cau_hinh as ch


def khoang_wilson(so_dung, tong, z=None):
    """Khoảng tin cậy Wilson cho tỷ lệ, ổn định hơn công thức chuẩn thông
    thường khi tỷ lệ gần 0 hoặc gần 1, và khi số mẫu nhỏ."""
    if z is None:
        z = ch.MUC_TIN_CAY
    if tong == 0:
        return (0.0, 0.0)
    p = so_dung / tong
    mau_so = 1 + z * z / tong
    tam = p + z * z / (2 * tong)
    lech = z * math.sqrt(p * (1 - p) / tong + z * z / (4 * tong * tong))
    return ((tam - lech) / mau_so, (tam + lech) / mau_so)


def do_ket_qua(ten_mo_hinh, y_that, y_doan, giay=None, in_ra=True):
    acc = accuracy_score(y_that, y_doan)
    thap, cao = khoang_wilson(int(round(acc * len(y_that))), len(y_that))
    r = {
        "so_mau_test": len(y_that),
        "accuracy": round(float(acc), 4),
        "wilson_thap": round(thap, 4),
        "wilson_cao": round(cao, 4),
        "macro_f1": round(float(f1_score(y_that, y_doan, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_that, y_doan, average="weighted")), 4),
    }
    if giay is not None:
        r["giay_huan_luyen"] = round(giay, 1)
    if in_ra:
        print(ten_mo_hinh)
        print("   accuracy    :", r["accuracy"],
              "(Wilson 95:", r["wilson_thap"], "đến", r["wilson_cao"], ")")
        print("   macro-F1    :", r["macro_f1"])
        print("   weighted-F1 :", r["weighted_f1"])
        if giay is not None:
            print("   thời gian   :", r["giay_huan_luyen"], "giây")
        print()
    return r


def bao_cao_tung_lop(y_that, y_doan, nhan_lop):
    print("BÁO CÁO CHI TIẾT TỪNG LỚP")
    print(classification_report(y_that, y_doan, target_names=nhan_lop,
                                digits=4, zero_division=0))


def ve_ma_tran_nham_lan(y_that, y_doan, nhan_lop, ten_mo_hinh, luu_hinh=True):
    """Vẽ và lưu ma trận nhầm lẫn, đồng thời in ra các cặp bị nhầm nhiều nhất."""
    import matplotlib
    if luu_hinh:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_that, y_doan, labels=nhan_lop)
    ve = [ch.TEN_HIEN_THI.get(n, n) for n in nhan_lop]

    fig, ax = plt.subplots(figsize=(9, 8))
    hinh = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(len(nhan_lop)))
    ax.set_yticks(np.arange(len(nhan_lop)))
    ax.set_xticklabels(ve, rotation=45, ha="right")
    ax.set_yticklabels(ve)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Ma trận nhầm lẫn _ " + ten_mo_hinh)
    nguong = cm.max() / 2
    for i in range(len(nhan_lop)):
        for j in range(len(nhan_lop)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > nguong else "black", fontsize=8)
    fig.colorbar(hinh)
    plt.tight_layout()

    goc = os.path.join(ch.THU_MUC_KET_QUA, "ma_tran_nham_lan_" + ten_mo_hinh)
    plt.savefig(goc + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(cm, index=ve, columns=ve).to_csv(goc + ".csv", encoding="utf-8")
    print("Đã lưu", goc + ".png", "và", goc + ".csv")
    print()

    cap = []
    for i in range(len(nhan_lop)):
        for j in range(len(nhan_lop)):
            if i != j and cm[i, j] > 0:
                cap.append((int(cm[i, j]), ve[i], ve[j]))
    cap.sort(reverse=True)
    print("MƯỜI CẶP BỊ NHẦM NHIỀU NHẤT (nhãn thật _ bị đoán thành _ số lượng)")
    for so, that, doan in cap[:10]:
        print("   ", that, "_", doan, "_", so)
    print()
    return cm


def ve_phan_bo_nhan(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bang = df.pivot_table(index="chu_de", columns="tap", values="so_tu",
                          aggfunc="count").fillna(0).astype(int)
    bang = bang[["train", "test"]]
    bang["tong"] = bang["train"] + bang["test"]
    sap = bang.sort_values("tong", ascending=False)
    ve = [ch.TEN_HIEN_THI.get(i, i) for i in sap.index]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    vi_tri = np.arange(len(sap))
    ax[0].bar(vi_tri - 0.2, sap["train"], width=0.4, label="Train")
    ax[0].bar(vi_tri + 0.2, sap["test"], width=0.4, label="Test")
    ax[0].set_xticks(vi_tri)
    ax[0].set_xticklabels(ve, rotation=45, ha="right")
    ax[0].set_ylabel("Số văn bản")
    ax[0].set_title("Phân bố nhãn theo tập")
    ax[0].legend()
    ax[1].barh(ve[::-1], sap["tong"][::-1])
    ax[1].set_xlabel("Tổng số văn bản")
    ax[1].set_title("Tổng số văn bản mỗi chủ đề")
    plt.tight_layout()
    p = os.path.join(ch.THU_MUC_KET_QUA, "bieu_do_phan_bo_nhan.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Đã lưu", p)
    return bang


def luu_ket_qua(ten_mo_hinh, ket_qua, khoa="ket_qua_test"):
    """Gộp vào tệp kết quả chung để chạy nhiều lượt vẫn ra một bảng đủ."""
    tat_ca = {}
    if os.path.exists(ch.TEP_KET_QUA):
        with open(ch.TEP_KET_QUA, encoding="utf-8") as f:
            tat_ca = json.load(f)
    tat_ca.setdefault(khoa, {})[ten_mo_hinh] = ket_qua
    with open(ch.TEP_KET_QUA, "w", encoding="utf-8") as f:
        json.dump(tat_ca, f, ensure_ascii=False, indent=1)


def khoa_lan_chay_moi(ten_goc, khoa="ket_qua_test"):
    """Sinh khóa cho một lần chạy mới, dạng ten_goc_lan1, ten_goc_lan2 ...

    Cần cho PhoBERT vì mô hình này không tất định, ghi đè thì chỉ giữ được lần
    chạy gần nhất. Ba mô hình cổ điển tất định nên vẫn ghi đè bình thường.
    """
    tat_ca = doc_ket_qua().get(khoa, {})
    n = 1
    while ten_goc + "_lan" + str(n) in tat_ca:
        n += 1
    return ten_goc + "_lan" + str(n)


def gom_cac_lan_chay(kq):
    """Gom các khóa dạng ten_goc_lanN lại và tính trung bình kèm độ lệch chuẩn."""
    nhom = {}
    for ten, r in kq.items():
        if "_lan" in ten:
            goc = ten.rsplit("_lan", 1)[0]
            nhom.setdefault(goc, []).append(r)
    tong_hop = {}
    for goc, ds in nhom.items():
        if len(ds) < 2:
            continue
        tong_hop[goc] = {
            "so_lan": len(ds),
            "acc_tb": round(float(np.mean([d["accuracy"] for d in ds])), 4),
            "acc_lech": round(float(np.std([d["accuracy"] for d in ds], ddof=1)), 4),
            "f1_tb": round(float(np.mean([d["macro_f1"] for d in ds])), 4),
            "f1_lech": round(float(np.std([d["macro_f1"] for d in ds], ddof=1)), 4),
        }
    return tong_hop


def doc_ket_qua():
    if not os.path.exists(ch.TEP_KET_QUA):
        return {}
    with open(ch.TEP_KET_QUA, encoding="utf-8") as f:
        return json.load(f)


def in_bang_tong_hop(nhan_lop=None, y_that=None, du_doan_luu=None):
    """In bảng tổng hợp để dán vào báo cáo."""
    tat_ca = doc_ket_qua()
    kq = tat_ca.get("ket_qua_test", {})
    cv = tat_ca.get("ket_qua_cv", {})

    print("=" * 78)
    print("BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 78)
    print()

    if os.path.exists(ch.TEP_THONG_KE):
        with open(ch.TEP_THONG_KE, encoding="utf-8") as f:
            tk = json.load(f)
        print("A. DỮ LIỆU")
        print("   Kho ngữ liệu       : VNTC 10Topics Ver1.1, github.com/duyvuleo/VNTC")
        for k, v in tk.items():
            print("   {:<21}:".format(k), v)
        print()

    if kq:
        print("B. KẾT QUẢ TRÊN TẬP KIỂM THỬ")
        print()
        dong = "{:<24}{:>10}{:>22}{:>11}{:>13}{:>9}"
        print(dong.format("Mô hình", "Accuracy", "Wilson 95", "macro-F1",
                          "weighted-F1", "Giây"))
        print("-" * 89)
        for ten in sorted(kq):
            r = kq[ten]
            print(dong.format(ten, r["accuracy"],
                              str(r["wilson_thap"]) + " _ " + str(r["wilson_cao"]),
                              r["macro_f1"], r["weighted_f1"],
                              r.get("giay_huan_luyen", "")))
        print()

    nhieu_lan = gom_cac_lan_chay(kq)
    if nhieu_lan:
        print("B2. MÔ HÌNH CHẠY NHIỀU LẦN, TRUNG BÌNH KÈM ĐỘ LỆCH CHUẨN")
        print()
        d3 = "{:<26}{:>8}{:>24}{:>24}"
        print(d3.format("Mô hình", "Số lần", "Accuracy", "macro-F1"))
        print("-" * 82)
        for goc in sorted(nhieu_lan):
            r = nhieu_lan[goc]
            print(d3.format(goc, r["so_lan"],
                            str(r["acc_tb"]) + " +/- " + str(r["acc_lech"]),
                            str(r["f1_tb"]) + " +/- " + str(r["f1_lech"])))
        print()
        print("   Đây là con số nên đưa vào báo cáo cho mô hình không tất định,")
        print("   thay vì lấy kết quả của một lần chạy đơn lẻ.")
        print()

    if cv:
        print("C. KIỂM ĐỊNH CHÉO 5 LỚP (chạy trên tập huấn luyện)")
        print()
        dong2 = "{:<24}{:>22}{:>24}"
        print(dong2.format("Mô hình", "Accuracy trung bình", "macro-F1 trung bình"))
        print("-" * 70)
        for ten in sorted(cv):
            r = cv[ten]
            print(dong2.format(ten,
                               str(r["acc_tb"]) + " +/- " + str(r["acc_lech"]),
                               str(r["f1_tb"]) + " +/- " + str(r["f1_lech"])))
        print()

    if kq:
        tot = max(kq, key=lambda k: kq[k]["macro_f1"])
        print("D. MÔ HÌNH TỐT NHẤT THEO macro-F1:", tot)
        print("   macro-F1 =", kq[tot]["macro_f1"])
        print()

        if du_doan_luu and tot in du_doan_luu and y_that is not None:
            print("E. F1 TỪNG LỚP CỦA MÔ HÌNH TỐT NHẤT")
            bc = classification_report(y_that, du_doan_luu[tot],
                                       target_names=nhan_lop,
                                       output_dict=True, zero_division=0)
            for n in nhan_lop:
                print("   {:<20} F1 = {:.4f}   số mẫu = {}".format(
                    ch.TEN_HIEN_THI.get(n, n), bc[n]["f1-score"], int(bc[n]["support"])))
            print()

    print("=" * 78)
    print("CÁC TỆP KẾT QUẢ ĐÃ SINH RA:")
    if os.path.isdir(ch.THU_MUC_KET_QUA):
        for t in sorted(os.listdir(ch.THU_MUC_KET_QUA)):
            p = os.path.join(ch.THU_MUC_KET_QUA, t)
            if os.path.isfile(p):
                print("   ", t, "_", round(os.path.getsize(p) / 1024, 1), "KB")
