# Phân loại chủ đề bài viết tiếng Việt trên kho ngữ liệu VNTC

Đề tài xây dựng và đối chứng bốn mô hình phân loại chủ đề cho văn bản tin tức
tiếng Việt, gồm ba mô hình cổ điển trên đặc trưng TF-IDF và một mô hình tinh
chỉnh PhoBERT, trong cùng một điều kiện thực nghiệm.


Có ba cách chạy dự án, trình bày lần lượt ở ba mục dưới.

## . Cách một: chạy trực tiếp bằng Python


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

Muốn chạy thử nhanh với một phần dữ liệu thì thêm `--mau_thu`. Bước `chuan_bi`
lưu kết quả tách từ ra `data/du_lieu_da_tach_tu.csv`, các lần chạy sau tự đọc lại
tệp này nên không phải tách từ lại. Muốn làm lại từ đầu thì thêm `--lam_lai`.

## 3. Cách hai: chạy API và giao diện web thủ công

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

## 4. Cách ba: chạy bằng Docker

```
docker compose up --build
```

Sau khi khởi động xong, truy cập:

| Địa chỉ | Nội dung |
|---|---|
| http://localhost:8080 | Giao diện web |
| http://localhost:8000/docs | Trang tài liệu API tương tác |
| http://localhost:8000/health | Kiểm tra dịch vụ và mô hình |

Dừng lại bằng tổ hợp phím Ctrl và C, hoặc bằng lệnh `docker compose down`.

## 5. Kết quả đã đo được

Chạy trên toàn bộ dữ liệu sau khi khử trùng lặp, tập kiểm thử 46.788 văn bản:

| Mô hình | Accuracy | Wilson 95 | macro-F1 | weighted-F1 | Giây |
|---|---|---|---|---|---|
| Naive Bayes | 0,9055 | 0,9028 _ 0,9081 | 0,8861 | 0,9053 | 0,3 |
| Logistic Regression | 0,9386 | 0,9364 _ 0,9407 | 0,9244 | 0,9382 | 52,4 |
| Linear SVM | 0,9366 | 0,9344 _ 0,9388 | 0,9197 | 0,9357 | 6,9 |

Kiểm định chéo 5 lớp trên tập huấn luyện:

| Mô hình | Accuracy | macro-F1 |
|---|---|---|
| Naive Bayes | 0,8955 +/- 0,0043 | 0,8909 +/- 0,0037 |
| Logistic Regression | 0,9297 +/- 0,0028 | 0,9278 +/- 0,0024 |
| Linear SVM | 0,9342 +/- 0,0024 | 0,9323 +/- 0,0018 |

Ba điểm đáng lưu ý khi đọc bảng trên. Một là Logistic Regression nhỉnh hơn trên
tập kiểm thử nhưng Linear SVM lại nhỉnh hơn và ổn định hơn trong kiểm định chéo,
chênh lệch chỉ khoảng 0,2 điểm phần trăm nên kết luận trung thực là hai mô hình
ngang nhau. Hai là Linear SVM huấn luyện nhanh hơn Logistic Regression khoảng
7,6 lần mà cho kết quả tương đương. Ba là lớp yếu nhất là Đời sống, bị nhầm nhiều
nhất sang Văn hóa và Chính trị Xã hội, điều này hợp lý vì ranh giới ba chuyên mục
đó vốn mờ ngay cả với người biên tập.

## 5b. Chạy PhoBERT nhiều lần

PhoBERT không tất định: hai lần chạy cùng cấu hình, cùng dữ liệu, cùng hạt giống
vẫn cho kết quả khác nhau. Nguyên nhân là thứ tự trộn dữ liệu mỗi epoch, tính
không xác định của một số phép toán trên GPU, và cách cộng số thực ở chế độ fp16
phụ thuộc thứ tự. Ba mô hình cổ điển thì tất định nên chạy một lần là đủ.

Vì vậy mỗi lần chạy `--buoc phobert` được lưu dưới một khóa riêng, dạng
`M4_PhoBERT_head_lan1`, `M4_PhoBERT_head_lan2` và cứ thế, thay vì ghi đè lên
nhau. Chỉ cần chạy lệnh dưới đây ba lần liên tiếp:

```
python chay.py --buoc phobert
python chay.py --buoc phobert
python chay.py --buoc phobert
python chay.py --buoc tong_hop
```

## 6. Về nhánh PhoBERT
Trước khi chạy, cài thêm:

```
pip install torch transformers
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```
