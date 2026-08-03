
import os
import sys

# Cho phép import gói src khi chạy từ thư mục gốc hoặc từ trong container
GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GOC not in sys.path:
    sys.path.insert(0, GOC)

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import cau_hinh as ch
from src.du_doan import BoPhanLoai

ung_dung = FastAPI(
    title="Phân loại chủ đề bài viết tiếng Việt",
    description="Báo cáo cuối kỳ môn Xử lý ngôn ngữ tự nhiên, lớp N01 nhóm 25",
    version="1.0",
)

# Cho phép trang frontend gọi sang API. Môi trường thật nên giới hạn allow_origins.
ung_dung.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp mô hình một lần lúc khởi động, không nạp lại theo từng yêu cầu
_bo_phan_loai = None


def lay_bo_phan_loai():
    global _bo_phan_loai
    if _bo_phan_loai is None:
        _bo_phan_loai = BoPhanLoai()
    return _bo_phan_loai


class YeuCauMot(BaseModel):
    van_ban: str
    top_k: int = 3


class YeuCauNhieu(BaseModel):
    danh_sach: List[str]
    top_k: int = 3


@ung_dung.get("/")
def goc():
    return {"thong_bao": "Dịch vụ phân loại chủ đề tiếng Việt đang chạy",
            "tai_lieu": "/docs"}


@ung_dung.get("/health")
def kiem_tra_suc_khoe():
    """Kiểm tra dịch vụ còn sống và mô hình đã nạp được chưa."""
    try:
        bpl = lay_bo_phan_loai()
        return {"trang_thai": "san_sang", "so_lop": len(bpl.nhan_lop),
                "cac_lop": [ch.TEN_HIEN_THI.get(n, n) for n in bpl.nhan_lop]}
    except FileNotFoundError as e:
        return {"trang_thai": "chua_co_mo_hinh", "chi_tiet": str(e)}


@ung_dung.post("/predict")
def du_doan_mot(yc: YeuCauMot):
    """Dự đoán chủ đề cho một văn bản."""
    if not yc.van_ban.strip():
        raise HTTPException(status_code=400, detail="Văn bản rỗng")
    try:
        bpl = lay_bo_phan_loai()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    kq = bpl.du_doan(yc.van_ban, top_k=yc.top_k)
    return {"ket_qua": kq, "nhan_cao_nhat": kq[0]["ten_hien_thi"]}


@ung_dung.post("/predict_batch")
def du_doan_nhieu(yc: YeuCauNhieu):
    """Dự đoán chủ đề cho nhiều văn bản trong một lần gọi."""
    if not yc.danh_sach:
        raise HTTPException(status_code=400, detail="Danh sách rỗng")
    try:
        bpl = lay_bo_phan_loai()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"ket_qua": bpl.du_doan_nhieu(yc.danh_sach, top_k=yc.top_k)}
