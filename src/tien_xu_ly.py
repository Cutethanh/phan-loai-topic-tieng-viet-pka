"""Tải dữ liệu, đọc tệp, khử trùng lặp và tách từ tiếng Việt
"""

import collections
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata

import pandas as pd

from . import cau_hinh as ch

MAU_TEN_TEP = re.compile(r"^([A-Za-z]+)_\s*([A-Za-z]+)_")


def ghi(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


# TẢI DỮ LIỆU
def tai_du_lieu():
    """Tải kho ngữ liệu từ GitHub và giải nén."""
    ch.bao_dam_thu_muc()
    thu_muc_kho = os.path.join(ch.THU_MUC_DU_LIEU, "VNTC")

    if os.path.exists(os.path.join(ch.THU_MUC_VNTC, "Train_Full")):
        ghi("Dữ liệu đã được giải nén sẵn, bỏ qua bước tải")
        return

    if not os.path.exists(thu_muc_kho):
        ghi("Đang tải kho ngữ liệu từ GitHub, khoảng 160 MB")
        subprocess.run(["git", "clone", "--depth", "1", "-q", ch.URL_KHO, thu_muc_kho],
                       check=True)

    os.makedirs(ch.THU_MUC_VNTC, exist_ok=True)
    for ten in ["Train_Full.rar", "Test_Full.rar"]:
        nguon = os.path.join(thu_muc_kho, "Data", "10Topics", "Ver1.1", ten)
        ghi("Đang giải nén " + ten)
        giai_nen_rar(nguon, ch.THU_MUC_VNTC)
    ghi("Giải nén xong")


def giai_nen_rar(tep_rar, thu_muc_ra):
    """Giải nén tệp .rar, thử lần lượt nhiều công cụ để chạy được trên nhiều hệ
    điều hành. Định dạng .rar là độc quyền nên Python không có thư viện thuần."""
    cac_lenh = [
        ["unar", "-q", "-o", thu_muc_ra, tep_rar],   # Linux, macOS
        ["tar", "-xf", tep_rar, "-C", thu_muc_ra],   # Windows 10 trở lên
        ["7z", "x", "-y", "-o" + thu_muc_ra, tep_rar],
        ["bsdtar", "-xf", tep_rar, "-C", thu_muc_ra],
    ]
    for lenh in cac_lenh:
        try:
            kq = subprocess.run(lenh, capture_output=True)
        except FileNotFoundError:
            continue
        if kq.returncode == 0:
            ghi("   giải nén bằng " + lenh[0])
            return True

    raise RuntimeError


# ĐỌC VÀ CHUẨN HÓA

def doc_mot_tep(duong_dan):
    """Đọc một tệp văn bản của VNTC và trả về chuỗi đã chuẩn hóa."""
    with open(duong_dan, "rb") as f:
        raw = f.read()
    # Bắt buộc utf-16, không được để encoding mặc định
    van_ban = raw.decode("utf-16", errors="replace")
    # NFC để cùng một chữ không sinh ra hai đặc trưng khác nhau
    van_ban = unicodedata.normalize("NFC", van_ban)
    return " ".join(van_ban.split())


def tach_ma_toa_soan(ten_tep):
    m = MAU_TEN_TEP.match(ten_tep)
    if m:
        ma = m.group(2).upper()
        if ma in ch.MA_TOA_SOAN:
            return ma
    return "KHAC"


def quet_toan_bo():
    """Duyệt toàn bộ cây thư mục và đọc mọi tệp .txt."""
    ban_ghi = []
    for tap in ["Train_Full", "Test_Full"]:
        goc = os.path.join(ch.THU_MUC_VNTC, tap)
        if not os.path.isdir(goc):
            raise FileNotFoundError("Không thấy thư mục " + goc +
                                    ". Hãy chạy bước tải dữ liệu trước.")
        for chu_de in sorted(os.listdir(goc)):
            thu_muc_lop = os.path.join(goc, chu_de)
            if not os.path.isdir(thu_muc_lop):
                continue
            for ten in sorted(os.listdir(thu_muc_lop)):
                if not ten.lower().endswith(".txt"):
                    continue
                vb = doc_mot_tep(os.path.join(thu_muc_lop, ten))
                ban_ghi.append({
                    "tap": "train" if tap == "Train_Full" else "test",
                    "chu_de": chu_de,
                    "ten_tep": ten,
                    "toa_soan": tach_ma_toa_soan(ten),
                    "so_tu": len(vb.split()),
                    "van_ban": vb,
                    "bam": hashlib.md5(vb.lower().encode("utf-8")).hexdigest(),
                })
    return ban_ghi


# KHỬ TRÙNG LẶP
def khu_trung_lap(ban_ghi):
    
    nhom = collections.defaultdict(list)
    for i, r in enumerate(ban_ghi):
        nhom[r["bam"]].append(i)

    bo_di = set()
    so_nhom_mau_thuan = 0
    so_tep_mau_thuan = 0
    so_trung_train_test = 0

    for _, chi_so in nhom.items():
        nhan = {ban_ghi[i]["chu_de"] for i in chi_so}
        if len(nhan) > 1:
            so_nhom_mau_thuan += 1
            so_tep_mau_thuan += len(chi_so)
            bo_di.update(chi_so)
            continue
        if len(chi_so) == 1:
            continue
        cac_tap = {ban_ghi[i]["tap"] for i in chi_so}
        if "train" in cac_tap and "test" in cac_tap:
            so_trung_train_test += 1
        giu = None
        for i in chi_so:
            if ban_ghi[i]["tap"] == "train":
                giu = i
                break
        if giu is None:
            giu = chi_so[0]
        bo_di.update(set(chi_so) - {giu})

    con_lai = [r for i, r in enumerate(ban_ghi) if i not in bo_di]
    thong_ke = {
        "tong_ban_dau": len(ban_ghi),
        "so_tep_bi_loai": len(bo_di),
        "so_nhom_nhan_mau_thuan": so_nhom_mau_thuan,
        "so_tep_thuoc_nhom_mau_thuan": so_tep_mau_thuan,
        "so_noi_dung_trung_ca_train_va_test": so_trung_train_test,
        "tong_con_lai": len(con_lai),
    }
    return con_lai, thong_ke


# TÁCH TỪ

_bo_tach_tu = None


def _nap_bo_tach_tu():
    global _bo_tach_tu
    if _bo_tach_tu is None:
        from pyvi import ViTokenizer
        _bo_tach_tu = ViTokenizer
    return _bo_tach_tu


def tien_xu_ly(van_ban):
    """Chuỗi tiền xử lý dùng chung cho cả lúc huấn luyện và lúc dự đoán.

    Viết một lần và gọi ở cả hai nơi là bắt buộc: nếu lúc triển khai bỏ sót bước
    tách từ thì độ chính xác sụt mạnh mà không có thông báo lỗi nào.
    """
    tok = _nap_bo_tach_tu()
    van_ban = unicodedata.normalize("NFC", van_ban)
    van_ban = " ".join(van_ban.split())
    return tok.tokenize(van_ban)


def tach_tu_hang_loat(ban_ghi):
    tok = _nap_bo_tach_tu()
    tong = len(ban_ghi)
    t0 = time.time()
    moc = 10000
    for i, r in enumerate(ban_ghi):
        r["van_ban_tach_tu"] = tok.tokenize(r["van_ban"])
        if i + 1 >= moc:
            ghi("   đã tách từ " + str(i + 1) + " / " + str(tong)
                + " sau " + str(round(time.time() - t0, 1)) + " giây")
            moc += 10000
    ghi("Tách từ xong sau " + str(round(time.time() - t0, 1)) + " giây")
    return ban_ghi


# ĐIỀU PHỐI

def chuan_bi_du_lieu(mau_thu=False, so_mau_moi_lop=300, bat_buoc_lam_lai=False):
    """Chạy toàn bộ giai đoạn 1 và trả về khung dữ liệu đã tách từ.

    Nếu đã có tệp đã tách từ từ lần chạy trước thì đọc lại, bỏ qua tải và tách từ.
    """
    ch.bao_dam_thu_muc()

    if os.path.exists(ch.TEP_DA_TACH_TU) and not bat_buoc_lam_lai:
        ghi("Đọc lại dữ liệu đã tách từ từ " + ch.TEP_DA_TACH_TU)
        df = pd.read_csv(ch.TEP_DA_TACH_TU).dropna(subset=["van_ban_tach_tu"])
        ghi("Có " + str(len(df)) + " văn bản")
        return df

    tai_du_lieu()

    ghi("Đọc toàn bộ tệp văn bản")
    t0 = time.time()
    ban_ghi = quet_toan_bo()
    ghi("Đọc xong " + str(len(ban_ghi)) + " văn bản trong "
        + str(round(time.time() - t0, 1)) + " giây")

    ghi("Khử trùng lặp")
    ban_ghi, tk = khu_trung_lap(ban_ghi)
    for k, v in tk.items():
        ghi("   " + k + " = " + str(v))

    if mau_thu:
        dem = collections.Counter()
        loc = []
        for r in ban_ghi:
            khoa = (r["tap"], r["chu_de"])
            if dem[khoa] < so_mau_moi_lop:
                dem[khoa] += 1
                loc.append(r)
        ban_ghi = loc
        ghi("CHẾ ĐỘ CHẠY THỬ: chỉ giữ " + str(len(ban_ghi)) + " văn bản")

    ghi("Tách từ tiếng Việt bằng pyvi, đây là bước lâu nhất")
    ban_ghi = tach_tu_hang_loat(ban_ghi)

    df = pd.DataFrame([{k: r[k] for k in
                        ["tap", "chu_de", "ten_tep", "toa_soan", "so_tu", "van_ban_tach_tu"]}
                       for r in ban_ghi])
    df.to_csv(ch.TEP_DA_TACH_TU, index=False, encoding="utf-8")
    ghi("Đã lưu " + ch.TEP_DA_TACH_TU)

    with open(ch.TEP_THONG_KE, "w", encoding="utf-8") as f:
        json.dump(tk, f, ensure_ascii=False, indent=1)

    return df


def chia_tap(df):
    """Dùng bản chia chuẩn có sẵn của VNTC để báo cáo đối chiếu được với các
    nghiên cứu khác, nên không gọi train_test_split."""
    tr = df[df["tap"] == "train"].reset_index(drop=True)
    te = df[df["tap"] == "test"].reset_index(drop=True)
    nhan_lop = sorted(tr["chu_de"].unique())
    return tr, te, nhan_lop
