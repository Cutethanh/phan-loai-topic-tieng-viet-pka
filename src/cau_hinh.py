import os

#ĐƯỜNG DẪN

THU_MUC_GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THU_MUC_DU_LIEU = os.path.join(THU_MUC_GOC, "data")
THU_MUC_KET_QUA = os.path.join(THU_MUC_GOC, "ket_qua")
THU_MUC_MO_HINH = os.path.join(THU_MUC_GOC, "models")
THU_MUC_VNTC = os.path.join(THU_MUC_DU_LIEU, "vntc_ext")

# Tệp lưu văn bản đã tách từ. Tách từ là bước lâu nhất nên phải lưu để dùng lại.
TEP_DA_TACH_TU = os.path.join(THU_MUC_DU_LIEU, "du_lieu_da_tach_tu.csv")
TEP_THONG_KE = os.path.join(THU_MUC_KET_QUA, "thong_ke_du_lieu.json")
TEP_KET_QUA = os.path.join(THU_MUC_KET_QUA, "ket_qua.json")
TEP_MO_HINH = os.path.join(THU_MUC_MO_HINH, "mo_hinh_phan_loai.joblib")

URL_KHO = "https://github.com/duyvuleo/VNTC.git"

# Cố định để mọi lần chạy cho cùng kết quả
HAT_GIONG = 42

#NHÃN LỚP

# Tên thư mục trong bộ dữ liệu là không dấu nên phải ánh xạ sang tên có dấu
TEN_HIEN_THI = {
    "Chinh tri Xa hoi": "Chính trị Xã hội",
    "Doi song": "Đời sống",
    "Khoa hoc": "Khoa học",
    "Kinh doanh": "Kinh doanh",
    "Phap luat": "Pháp luật",
    "Suc khoe": "Sức khỏe",
    "The gioi": "Thế giới",
    "The thao": "Thể thao",
    "Van hoa": "Văn hóa",
    "Vi tinh": "Vi tính",
}

# Mã tòa soạn nằm ở TÊN TỆP chứ không nằm trong nội dung
MA_TOA_SOAN = {
    "VNE": "VnExpress",
    "TT": "Tuổi Trẻ",
    "TN": "Thanh Niên",
    "NLD": "Người Lao Động",
}

# THAM SỐ MÔ HÌNH

THAM_SO_TFIDF = {
    "ngram_range": (1, 2),    # bigram bù cho những từ ghép pyvi tách chưa chuẩn
    "min_df": 3,              # bỏ từ quá hiếm, thường là lỗi chính tả
    "max_df": 0.9,            # bỏ từ quá phổ biến, thay cho danh sách từ dừng
    "sublinear_tf": True,     # thay tf bằng 1 cộng logarit của tf
    "max_features": 300000,   # chặn trần bộ nhớ
}

PHOBERT = {
    "mo_hinh_goc": "vinai/phobert-base",
    "do_dai_toi_da": 256,     # giới hạn cứng của PhoBERT, không tăng được
    "batch_size": 16,         # hạ xuống 8 nếu báo lỗi hết bộ nhớ GPU
    "gop_dao_ham": 2,         # batch hiệu dụng bằng batch_size nhân gop_dao_ham
    "so_epoch": 3,
    "toc_do_hoc": 2e-5,
    "weight_decay": 0.01,
    "ty_le_warmup": 0.1,
    "cat_dao_ham": 1.0,
    "kieu_cat": "head",       # head lấy phần đầu, head_tail lấy đầu cộng cuối
}

SO_LOP_GAP = 5          # số lớp gấp của kiểm định chéo
MUC_TIN_CAY = 1.96      # hệ số z cho khoảng tin cậy 95


def bao_dam_thu_muc():
    for d in [THU_MUC_DU_LIEU, THU_MUC_KET_QUA, THU_MUC_MO_HINH]:
        os.makedirs(d, exist_ok=True)
