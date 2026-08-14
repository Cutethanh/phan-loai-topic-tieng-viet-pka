import os
import sys
GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GOC not in sys.path:
    sys.path.insert(0, GOC)
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src import cau_hinh as ch
from src.du_doan import BoPhanLoai

ung_dung = FastAPI(
    title="Phân loại chủ đề bài viết tiếng Việt",
    version="1.0",
)
ung_dung.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_bo_phan_loai = None
def lay_bo_phan_loai():
    global _bo_phan_loai
    if _bo_phan_loai is None:
        _bo_phan_loai = BoPhanLoai()
    return _bo_phan_loai
class YeuCauMot(BaseModel):
    van_ban: str
    top_k: int = Field(3, ge=1, le=10, description="So nhan tra ve, tu 1 toi 10")

class YeuCauNhieu(BaseModel):
    danh_sach: List[str]
    top_k: int = Field(3, ge=1, le=10, description="So nhan tra ve, tu 1 toi 10")

@ung_dung.get("/")
def goc():
    return {"thong_bao": "Dịch vụ phân loại chủ đề tiếng Việt đang chạy",
            "tai_lieu": "/docs"}

@ung_dung.get("/health")
def kiem_tra_suc_khoe():
    try:
        bpl = lay_bo_phan_loai()
        return {"trang_thai": "san_sang", "so_lop": len(bpl.nhan_lop),
                "cac_lop": [ch.TEN_HIEN_THI.get(n, n) for n in bpl.nhan_lop]}
    except FileNotFoundError as e:
        return {"trang_thai": "chua_co_mo_hinh", "chi_tiet": str(e)}

@ung_dung.post("/predict")
def du_doan_mot(yc: YeuCauMot):
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
    if any(not vb.strip() for vb in yc.danh_sach):
        raise HTTPException(status_code=400,
                            detail="Danh sách chứa văn bản rỗng")
    try:
        bpl = lay_bo_phan_loai()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"ket_qua": bpl.du_doan_nhieu(yc.danh_sach, top_k=yc.top_k)}
