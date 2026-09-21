# Phân loại chủ đề bài viết tiếng Việt trên kho ngữ liệu VNTC

Xây dựng và đối chứng bốn mô hình phân loại chủ đề cho văn bản tin tức
tiếng Việt, gồm ba mô hình cổ điển trên đặc trưng TF-IDF và một mô hình tinh
chỉnh PhoBERT, trong cùng một điều kiện thực nghiệm.


Có hai cách chạy dự án, trình bày lần lượt ở ba mục dưới.

## 1. Cách một: chạy trực tiếp bằng Python


```
pip install -r requirements.txt
```

Chạy toàn bộ nhánh cổ điển từ đầu tới cuối:

```
python chay.py --buoc tat_ca
```

Hoặc chạy từng bước một:

```
python chay.py --buoc chuan_bi     tai du lieu, khu trung lap, tach tu
python chay.py --buoc thong_ke     bang thong ke mo ta va bieu do
python chay.py --buoc co_dien      huan luyen va danh gia ba mo hinh
python chay.py --buoc kiem_dinh    kiem dinh cheo 5 lop
python chay.py --buoc phobert      tinh chinh PhoBERT, can GPU
python chay.py --buoc du_doan      thu du doan tren vai van ban mau
python chay.py --buoc tong_hop     in bang tong hop ket qua
```

