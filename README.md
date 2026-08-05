# Phân loại chủ đề bài viết tiếng Việt trên kho ngữ liệu VNTC

Xây dựng và đối chứng bốn mô hình phân loại chủ đề cho văn bản tin tức
tiếng Việt, gồm ba mô hình cổ điển trên đặc trưng TF-IDF và một mô hình tinh
chỉnh PhoBERT, trong cùng một điều kiện thực nghiệm.


Có ba cách chạy dự án, trình bày lần lượt ở ba mục dưới.

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
## 1.2. Về nhánh PhoBERT
Trước khi chạy, cài thêm:
```
pip install torch transformers
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python chay.py --buoc phobert 
```

Muốn chạy thử nhanh với một phần dữ liệu thì thêm `--mau_thu`.

## 2. Cách hai: chạy API và giao diện web thủ công

Phải chạy `python chay.py --buoc co_dien` trước để sinh ra tệp mô hình trong thư
mục `models/`.

Mở cửa sổ dòng lệnh thứ nhất, chạy backend:

```
uvicorn backend.main:ung_dung --reload --port 8000
```

Mở cửa sổ thứ hai, chạy một máy chủ tệp tĩnh cho frontend:

```
cd frontend
python -m http.server 8080
```

## 3. Cách ba: chạy bằng Docker

```
docker compose up --build
```

Sau khi khởi động xong, truy cập:

| Địa chỉ | Nội dung |
|---|---|
| http://localhost:8080 |web|

