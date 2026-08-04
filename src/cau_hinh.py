import os

#ĐƯỜNG DẪN

THU_MUC_GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THU_MUC_DU_LIEU = os.path.join(THU_MUC_GOC, "data")
THU_MUC_KET_QUA = os.path.join(THU_MUC_GOC, "ket_qua")
THU_MUC_MO_HINH = os.path.join(THU_MUC_GOC, "models")
THU_MUC_VNTC = os.path.join(THU_MUC_DU_LIEU, "vntc_ext")

TEP_DA_TACH_TU = os.path.join(THU_MUC_DU_LIEU, "du_lieu_da_tach_tu.csv")
TEP_THONG_KE = os.path.join(THU_MUC_KET_QUA, "thong_ke_du_lieu.json")
TEP_KET_QUA = os.path.join(THU_MUC_KET_QUA, "ket_qua.json")
TEP_MO_HINH = os.path.join(THU_MUC_MO_HINH, "mo_hinh_phan_loai.joblib")

URL_KHO = "https://github.com/duyvuleo/VNTC.git"

HAT_GIONG = 42

#NHÃN LỚP
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
MA_TOA_SOAN = {
    "VNE": "VnExpress",
    "TT": "Tuổi Trẻ",
    "TN": "Thanh Niên",
    "NLD": "Người Lao Động",
}
# THAM SỐ MÔ HÌNH
THAM_SO_TFIDF = {
    "ngram_range": (1, 2),    
    "min_df": 3,              
    "max_df": 0.9,            
    "sublinear_tf": True,     
    "max_features": 300000,   
}
PHOBERT = {
    "mo_hinh_goc": "vinai/phobert-base",
    "do_dai_toi_da": 256,     
    "batch_size": 16,         
    "gop_dao_ham": 2,        
    "so_epoch": 3,
    "toc_do_hoc": 2e-5,
    "weight_decay": 0.01,
    "ty_le_warmup": 0.1,
    "cat_dao_ham": 1.0,
    "kieu_cat": "head",       # head lấy phần đầu, head_tail lấy đầu cộng cuối
}

SO_LOP_GAP = 5          
MUC_TIN_CAY = 1.96      

def bao_dam_thu_muc():
    for d in [THU_MUC_DU_LIEU, THU_MUC_KET_QUA, THU_MUC_MO_HINH]:
        os.makedirs(d, exist_ok=True)
