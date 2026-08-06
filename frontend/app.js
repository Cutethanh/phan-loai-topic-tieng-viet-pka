const DIA_CHI_API = window.location.hostname
  ? "http://" + window.location.hostname + ":8000"
  : "http://127.0.0.1:8000";

const VAN_BAN_MAU = [
  "Đội tuyển Việt Nam giành chiến thắng 2-0 trong trận đấu tối qua tại sân Mỹ Đình, tiền đạo ghi bàn ở phút 78 sau đường chuyền của tiền vệ cánh phải.",
  "Giá cổ phiếu ngân hàng đồng loạt tăng trần, chỉ số VN-Index vượt mốc 1.200 điểm trong phiên giao dịch hôm nay với thanh khoản đạt mức cao nhất từ đầu năm.",
  "Các nhà khoa học vừa công bố phát hiện mới về cấu trúc protein giúp hiểu rõ hơn cơ chế phân chia tế bào, mở ra hướng nghiên cứu mới trong sinh học phân tử.",
  "Bộ Y tế khuyến cáo người dân tiêm nhắc lại vắc xin và rửa tay thường xuyên để phòng bệnh đường hô hấp trong mùa lạnh, đặc biệt là người cao tuổi.",
  "Bộ vi xử lý thế hệ mới hỗ trợ bộ nhớ DDR5 và có hiệu năng đồ họa tích hợp cao hơn đời trước khoảng ba mươi phần trăm theo công bố của nhà sản xuất.",
];

const oNhap = document.getElementById("o-nhap");
const nutDoan = document.getElementById("nut-doan");
const nutXoa = document.getElementById("nut-xoa");
const chonMau = document.getElementById("chon-mau");
const oKetQua = document.getElementById("ket-qua");
const oLoi = document.getElementById("loi");
const den = document.getElementById("den");
const oTrangThai = document.getElementById("trang-thai");
VAN_BAN_MAU.forEach((vb, i) => {
  const o = document.createElement("option");
  o.value = String(i);
  o.textContent = "Mẫu " + (i + 1) + ": " + vb.slice(0, 40) + " ...";
  chonMau.appendChild(o);
});

chonMau.addEventListener("change", () => {
  if (chonMau.value !== "") {
    oNhap.value = VAN_BAN_MAU[Number(chonMau.value)];
    chonMau.value = "";
  }
});

nutXoa.addEventListener("click", () => {
  oNhap.value = "";
  oKetQua.classList.add("an");
  oLoi.classList.add("an");
});

function hienLoi(thongDiep) {
  oLoi.textContent = thongDiep;
  oLoi.classList.remove("an");
  oKetQua.classList.add("an");
}

function hienKetQua(danhSach) {
  oLoi.classList.add("an");
  oKetQua.innerHTML = "";
  danhSach.forEach(r => {
    const dong = document.createElement("div");
    dong.className = "dong-kq";

    const ten = document.createElement("div");
    ten.className = "ten";
    ten.textContent = r.ten_hien_thi;

    const thanh = document.createElement("div");
    thanh.className = "thanh";
    const trong = document.createElement("div");
    trong.className = "thanh-trong";
    // Nhan voi 100 de doi ty le thanh phan tram be rong
    trong.style.width = Math.min(100, r.do_tin_cay * 100) + "%";
    thanh.appendChild(trong);

    const so = document.createElement("div");
    so.className = "so";
    so.textContent = r.do_tin_cay.toFixed(4);

    dong.appendChild(ten);
    dong.appendChild(thanh);
    dong.appendChild(so);
    oKetQua.appendChild(dong);
  });
  oKetQua.classList.remove("an");
}

async function kiemTraSucKhoe() {
  try {
    const ra = await fetch(DIA_CHI_API + "/health");
    const d = await ra.json();
    if (d.trang_thai === "san_sang") {
      den.className = "den den-tot";
      oTrangThai.textContent = "Dịch vụ sẵn sàng, mô hình nhận " + d.so_lop + " chủ đề";
      nutDoan.disabled = false;
    } else {
      den.className = "den den-hong";
      oTrangThai.textContent = "Chưa có mô hình. Hãy chạy lệnh python chay.py --buoc co_dien trước.";
      nutDoan.disabled = true;
    }
  } catch (e) {
    den.className = "den den-hong";
    oTrangThai.textContent = "Không kết nối được tới API ở " + DIA_CHI_API;
    nutDoan.disabled = true;
  }
}

nutDoan.addEventListener("click", async () => {
  const vb = oNhap.value.trim();
  if (!vb) {
    hienLoi("Hãy nhập một đoạn văn bản trước khi bấm phân loại.");
    return;
  }
  nutDoan.disabled = true;
  nutDoan.textContent = "Đang xử lý...";
  try {
    const ra = await fetch(DIA_CHI_API + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ van_ban: vb, top_k: 5 }),
    });
    if (!ra.ok) {
      const chiTiet = await ra.text();
      hienLoi("Máy chủ trả về lỗi " + ra.status + ". " + chiTiet);
      return;
    }
    const d = await ra.json();
    hienKetQua(d.ket_qua);
  } catch (e) {
    hienLoi("Lỗi kết nối: " + e.message);
  } finally {
    nutDoan.disabled = false;
    nutDoan.textContent = "Phân loại";
  }
});

kiemTraSucKhoe();
