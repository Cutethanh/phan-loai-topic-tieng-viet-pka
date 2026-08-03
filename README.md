# Phân loại chủ đề bài viết tiếng Việt trên kho ngữ liệu VNTC

Báo cáo cuối kỳ môn Xử lý ngôn ngữ tự nhiên.

Đề tài xây dựng và đối chứng bốn mô hình phân loại chủ đề cho văn bản tin tức
tiếng Việt, gồm ba mô hình cổ điển trên đặc trưng TF-IDF và một mô hình tinh
chỉnh PhoBERT, trong cùng một điều kiện thực nghiệm.

## 1. Cấu trúc thư mục

```
NLP_PhanLoaiChuDe/
    backend/                     dich vu API
        main.py                  ung dung FastAPI, ba diem cuoi
        Dockerfile
        requirements.txt
    frontend/                    giao dien web goi sang API
        index.html
        style.css
        app.js
        Dockerfile               nginx phuc vu tep tinh
    models/                      noi luu mo hinh da huan luyen
    src/                         ma nguon xu ly va huan luyen
        cau_hinh.py              toan bo tham so gom mot cho
        tien_xu_ly.py            tai, doc UTF-16, khu trung lap, tach tu
        dac_trung.py             TF-IDF kem canh bao ro ri du lieu
        mo_hinh_co_dien.py       Naive Bayes, Logistic Regression, Linear SVM
        mo_hinh_phobert.py       tinh chinh PhoBERT
        danh_gia.py              chi so, ma tran nham lan, bang tong hop
        du_doan.py               nap mo hinh va du doan van ban moi
    notebook/
        Nhom25_NLP_PhanLoaiChuDe.ipynb    ban chay tren Google Colab
    data/                        du lieu tai ve va tep da tach tu
    ket_qua/                     bieu do, ma tran nham lan, bang so lieu
    chay.py                      diem khoi chay bang dong lenh
    docker-compose.yml
    requirements.txt
    .dockerignore
    .gitignore
    README.md
```

Có ba cách chạy dự án, trình bày lần lượt ở ba mục dưới.

## 2. Cách một: chạy trực tiếp bằng Python

Cần Python 3.9 trở lên.

```
pip install -r requirements.txt
```

Trên Linux hoặc macOS dùng lệnh sau:

```
sudo apt-get install unar
```

Trên Windows thì không cần cài gì thêm nếu máy có lệnh tar sẵn có từ Windows 10.
Nếu không, hãy cài 7-Zip tại 7-zip.org, chương trình sẽ tự nhận ra và dùng.

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

Sau đó mở trình duyệt vào `http://127.0.0.1:8080` để dùng giao diện, hoặc vào
`http://127.0.0.1:8000/docs` để dùng trang tài liệu API tương tác. Trang tài liệu
này do FastAPI tự sinh, rất tiện khi demo trước hội đồng vì gõ thẳng văn bản vào
là thấy kết quả ngay.

## 4. Cách ba: chạy bằng Docker

Cần cài Docker Desktop. Vẫn phải huấn luyện mô hình trước để có tệp trong
`models/`, vì thư mục này được gắn vào container chứ không nằm trong ảnh.

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

## 5. Các điểm cuối của API

| Điểm cuối | Phương thức | Đầu vào | Đầu ra |
|---|---|---|---|
| /health | GET | không | Trạng thái dịch vụ và danh sách lớp |
| /predict | POST | một chuỗi văn bản | Nhãn chủ đề kèm độ tin cậy |
| /predict_batch | POST | danh sách chuỗi văn bản | Danh sách kết quả tương ứng |

Ví dụ gọi bằng curl:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\\"van_ban\\": \\"Doi tuyen Viet Nam gianh chien thang 2-0\\", \\"top_k\\": 3}"
```

## 6. Kết quả đã đo được

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

## 6b. Chạy PhoBERT nhiều lần

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

Bảng tổng hợp sẽ có thêm mục B2 in ra trung bình kèm độ lệch chuẩn của các lần
chạy. Đây mới là con số nên đưa vào báo cáo cho PhoBERT, thay vì lấy kết quả của
một lần chạy đơn lẻ.

## 7. Về nhánh PhoBERT

Mô đun `src/mo_hinh_phobert.py` chưa được chạy thử trong quá trình viết mã, vì
máy dùng để soạn mã nguồn không có GPU. Code được viết theo tài liệu chính thức
của thư viện transformers và của PhoBERT.

Trước khi chạy, cài thêm:

```
pip install torch transformers
```

Lưu ý riêng cho card dòng RTX 50: các card này dùng kiến trúc Blackwell với
compute capability sm_120. Những bản PyTorch dựng theo CUDA 12.1 hoặc 12.4 không
chứa nhân biên dịch cho kiến trúc này. Triệu chứng dễ gây hiểu nhầm là hàm
`torch.cuda.is_available` vẫn trả về True, nhưng tới lúc chạy thật thì báo lỗi
không tìm thấy nhân thực thi. Cách sửa:

```
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Lệnh `python chay.py --buoc phobert` tự kiểm tra môi trường GPU bằng một phép
nhân ma trận nhỏ trước khi bắt đầu huấn luyện, và dừng lại nếu phát hiện vấn đề.

Nếu báo lỗi hết bộ nhớ GPU, sửa trong `src/cau_hinh.py`: hạ `batch_size` từ 16
xuống 8 và tăng `gop_dao_ham` từ 2 lên 4, khi đó batch hiệu dụng vẫn là 32.

## 8. Ba điểm kỹ thuật đáng chú ý

Bảng mã. Các tệp của VNTC lưu bằng UTF-16 Little Endian có BOM, xuống dòng kiểu
CRLF. Mở bằng encoding mặc định là UTF-8 sẽ ra ký tự rác hoặc báo lỗi. Xem hàm
`doc_mot_tep` trong `src/tien_xu_ly.py`.

Trùng lặp dữ liệu. Bản chia chuẩn của bộ dữ liệu có 2.399 nội dung xuất hiện ở cả
tập huấn luyện lẫn tập kiểm thử, cùng 378 nhóm văn bản giống hệt nhau nhưng bị
gán hai nhãn khác nhau. Không xử lý thì mô hình được chấm điểm trên chính những
văn bản nó đã nhìn thấy. Sau khi lọc, dữ liệu còn 80.332 văn bản gồm 33.544 huấn
luyện và 46.788 kiểm thử. Xem hàm `khu_trung_lap` trong `src/tien_xu_ly.py`.

Rò rỉ khi tính TF-IDF. Gọi `fit_transform` trên toàn bộ dữ liệu rồi mới cắt ra
thành train và test là lỗi nghiêm trọng, vì hàm `fit` học bộ từ vựng và trọng số
idf từ dữ liệu nó nhìn thấy. Đoạn mã sai chạy trơn tru và không báo lỗi gì, đó là
lý do nó nguy hiểm. Xem chú thích đầu tệp `src/dac_trung.py`.

## 9. Nguồn dữ liệu và tài liệu tham khảo

Kho ngữ liệu VNTC phiên bản 10Topics Ver1.1 tại https://github.com/duyvuleo/VNTC

Hoang, C.D.V., Dinh, D., Nguyen, N.L., Ngo, Q.H. (2007). A Comparative Study on
Vietnamese Text Classification Methods. IEEE RIVF 2007, trang 267 tới 273.
https://ieeexplore.ieee.org/document/4223084/

Nguyen, D.Q., Nguyen, A.T. (2020). PhoBERT: Pre-trained language models for
Vietnamese. Findings of EMNLP 2020, trang 1037 tới 1042.
https://aclanthology.org/2020.findings-emnlp.92.pdf

Nguyen, D.-V., Nguyen, N.L.-T. (2023). Is word segmentation necessary for
Vietnamese sentiment classification. https://arxiv.org/abs/2301.00418

## 10. Thành viên nhóm

| Họ và tên | Mã sinh viên |
|---|---|
| Nguyễn Hoàng An | 24100126 |
| Hoàng Văn Đô | 24107948 |
| Nguyễn Danh Trung Thành | 24100505 |
