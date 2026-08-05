"""Tinh chỉnh PhoBERT cho bài toán phân loại chủ đề
"""

import math
import os
import time

import numpy as np

from . import cau_hinh as ch
from . import danh_gia as dg


def kiem_tra_thu_vien():
    thieu = []
    try:
        import torch  # noqa: F401
    except ImportError:
        thieu.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        thieu.append("transformers")
    if thieu:
        print("THIẾU THƯ VIỆN:", ", ".join(thieu))
        print("Cài bằng lệnh sau rồi chạy lại:")
        print("   pip install " + " ".join(thieu))
        print()
        print("Riêng với card dòng RTX 50 (kiến trúc Blackwell, sm_120), phải cài")
        print("bản torch theo CUDA 12.8 trở lên, nếu không sẽ báo lỗi không tìm")
        print("thấy nhân thực thi ngay cả khi đã nhận ra GPU:")
        print("   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128")
        return False
    return True

def kiem_tra_gpu():
    """Kiểm tra môi trường GPU và cảnh báo các vấn đề đã biết."""
    if not kiem_tra_thu_vien():
        return False
    import torch

    print("PyTorch     :", torch.__version__)
    print("Bản CUDA mà PyTorch được dựng theo:", torch.version.cuda)
    print("Thấy GPU    :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("KHÔNG THẤY GPU. Nhánh PhoBERT sẽ chạy rất chậm trên CPU.")
        return False

    ten = torch.cuda.get_device_name(0)
    kha_nang = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("Tên GPU     :", ten)
    print("Compute capability:", kha_nang)
    print("VRAM        :", round(vram, 1), "GB")

    if kha_nang[0] >= 12:
        print()
        print("LƯU Ý: GPU thuộc kiến trúc Blackwell (sm_120).")
        print("Nếu bước huấn luyện báo lỗi dạng no kernel image is available,")
        print("hãy cài lại PyTorch theo bản CUDA 12.8 trở lên.")

    try:
        a = torch.randn(64, 64, device="cuda")
        _ = a @ a
        torch.cuda.synchronize()
        print("Thử tính trên GPU: THÀNH CÔNG, GPU dùng được")
    except Exception as e:
        print("Thử tính trên GPU: THẤT BẠI")
        print("   ", type(e).__name__, str(e)[:200])
        print("   Đây gần như chắc chắn là vấn đề phiên bản PyTorch nêu ở trên.")
        return False

    if vram < 9:
        print("Với VRAM dưới 9 GB, nên giữ batch_size = 16 hoặc hạ xuống 8.")
    return True

def do_token_con_tren_tu(tokenizer, tr, so_mau=2000):
    mau = tr["van_ban_tach_tu"].sample(n=min(so_mau, len(tr)),
                                       random_state=ch.HAT_GIONG)
    tong_tu = 0
    tong_token = 0
    for vb in mau:
        tong_tu += len(str(vb).split())
        tong_token += len(tokenizer.encode(str(vb), add_special_tokens=False))
    r = tong_token / max(tong_tu, 1)
    con = ch.PHOBERT["do_dai_toi_da"] - 2
    print("Đo trên", len(mau), "văn bản của tập huấn luyện:")
    print("   Số token con trung bình trên một từ (r) =", round(r, 4))
    print("   Số vị trí dành cho nội dung =", con)
    print("   Số từ giữ lại được trung bình =", round(con / r, 1))
    print("   Trong khi độ dài trung vị của văn bản là 361 từ.")
    return r

def cat_chuoi(ids, kieu, id_dau, id_cuoi):
    con = ch.PHOBERT["do_dai_toi_da"] - 2
    if len(ids) <= con:
        giu = ids
    elif kieu == "head_tail":
        nua = con // 2
        giu = ids[:nua] + ids[len(ids) - (con - nua):]
    else:
        giu = ids[:con]
    return [id_dau] + giu + [id_cuoi]


def ma_hoa(khung, tokenizer, nhan_to_so, kieu):
    import torch
    from torch.utils.data import TensorDataset
    n = len(khung)
    L = ch.PHOBERT["do_dai_toi_da"]
    id_dem = tokenizer.pad_token_id
    X = np.full((n, L), id_dem, dtype=np.int64)
    M = np.zeros((n, L), dtype=np.int64)
    moc = 10000
    for i, vb in enumerate(khung["van_ban_tach_tu"].tolist()):
        ids = tokenizer.encode(str(vb), add_special_tokens=False)
        ids = cat_chuoi(ids, kieu, tokenizer.cls_token_id, tokenizer.sep_token_id)
        X[i, :len(ids)] = ids
        M[i, :len(ids)] = 1
        if i + 1 >= moc:
            print("   mã hóa", i + 1, "/", n)
            moc += 10000
    y = np.array([nhan_to_so[c] for c in khung["chu_de"].tolist()], dtype=np.int64)
    return TensorDataset(torch.tensor(X), torch.tensor(M), torch.tensor(y))

def tinh_chinh(tr, te, nhan_lop, kieu_cat=None):
    """Tinh chỉnh PhoBERT rồi đánh giá trên tập kiểm thử."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    P = ch.PHOBERT
    kieu_cat = kieu_cat or P["kieu_cat"]
    torch.manual_seed(ch.HAT_GIONG)
    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(P["mo_hinh_goc"])
    print("Đã nạp bộ tách token của", P["mo_hinh_goc"])
    do_token_con_tren_tu(tokenizer, tr)
    print()

    nhan_to_so = {n: i for i, n in enumerate(nhan_lop)}
    print("Mã hóa tập huấn luyện")
    ds_train = ma_hoa(tr, tokenizer, nhan_to_so, kieu_cat)
    print("Mã hóa tập kiểm thử")
    ds_test = ma_hoa(te, tokenizer, nhan_to_so, kieu_cat)

    dl_train = DataLoader(ds_train, batch_size=P["batch_size"], shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=P["batch_size"] * 2, shuffle=False)

    mo_hinh = AutoModelForSequenceClassification.from_pretrained(
        P["mo_hinh_goc"], num_labels=len(nhan_lop)).to(thiet_bi)

    opt = torch.optim.AdamW(mo_hinh.parameters(), lr=P["toc_do_hoc"],
                            weight_decay=P["weight_decay"])
    so_buoc_moi_epoch = math.ceil(len(dl_train) / P["gop_dao_ham"])
    tong_buoc = so_buoc_moi_epoch * P["so_epoch"]
    lich = get_linear_schedule_with_warmup(
        opt, int(P["ty_le_warmup"] * tong_buoc), tong_buoc)
    dung_fp16 = (thiet_bi == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dung_fp16)

    print()
    print("Bắt đầu tinh chỉnh")
    print("   Thiết bị             :", thiet_bi)
    print("   Batch size           :", P["batch_size"], "| gộp đạo hàm",
          P["gop_dao_ham"], "| batch hiệu dụng", P["batch_size"] * P["gop_dao_ham"])
    print("   Số epoch             :", P["so_epoch"])
    print("   Tổng số bước cập nhật:", tong_buoc)
    print()

    t0 = time.time()
    for epoch in range(P["so_epoch"]):
        mo_hinh.train()
        tong_loss = 0.0
        opt.zero_grad(set_to_none=True)
        dem_gop = 0
        moc_in = 200
        for buoc, (x, m, y) in enumerate(dl_train):
            x, m, y = x.to(thiet_bi), m.to(thiet_bi), y.to(thiet_bi)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dung_fp16):
                ra = mo_hinh(input_ids=x, attention_mask=m, labels=y)
                loss = ra.loss / P["gop_dao_ham"]
            scaler.scale(loss).backward()
            tong_loss += ra.loss.item()

            dem_gop += 1
            if dem_gop >= P["gop_dao_ham"] or (buoc + 1) == len(dl_train):
                dem_gop = 0
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), P["cat_dao_ham"])
                scaler.step(opt)
                scaler.update()
                lich.step()
                opt.zero_grad(set_to_none=True)

            if buoc + 1 >= moc_in:
                print("   epoch", epoch + 1, "batch", buoc + 1, "/", len(dl_train),
                      "| loss trung bình", round(tong_loss / (buoc + 1), 4),
                      "|", round(time.time() - t0, 1), "giây")
                moc_in += 200

        print("Xong epoch", epoch + 1, "| loss trung bình",
              round(tong_loss / len(dl_train), 4))
        diem_dung = os.path.join(ch.THU_MUC_KET_QUA, "phobert_epoch_" + str(epoch + 1))
        mo_hinh.save_pretrained(diem_dung)
        tokenizer.save_pretrained(diem_dung)
        print("   Đã lưu điểm dừng", diem_dung)
        print()

    giay = time.time() - t0
    print("Tinh chỉnh xong sau", round(giay / 60, 1), "phút")

    mo_hinh.eval()
    gom = []
    with torch.no_grad():
        for x, m, y in dl_test:
            x, m = x.to(thiet_bi), m.to(thiet_bi)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dung_fp16):
                logits = mo_hinh(input_ids=x, attention_mask=m).logits
            gom.append(logits.float().argmax(dim=-1).cpu().numpy())
    y_doan = np.array([nhan_lop[i] for i in np.concatenate(gom)])
    y_that = te["chu_de"].tolist()

    ten = dg.khoa_lan_chay_moi("M4_PhoBERT_" + kieu_cat)
    kq = dg.do_ket_qua(ten, y_that, y_doan, giay)
    dg.luu_ket_qua(ten, kq)
    dg.bao_cao_tung_lop(y_that, y_doan, nhan_lop)
    dg.ve_ma_tran_nham_lan(y_that, y_doan, nhan_lop, ten)
    return kq, y_doan, mo_hinh, tokenizer
