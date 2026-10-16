# BDD100K Pool Generator (RoadSift P-021)

Công cụ trích xuất dữ liệu BDD100K phục vụ quy trình Active Learning trên **RoadSift (P-021 / RAV-05)**.

---

## 🚀 Khởi chạy

Nhấp đúp **`run.bat`** hoặc chạy lệnh:
```bash
streamlit run streamlit_app.py
```

---

## 📌 4 Chức năng chính

### 1. Tạo Pool (Train)
* Rút ngẫu nhiên ảnh từ `train/images`, tự động loại trừ sequence thuộc Holdout (chống data leakage).
* Xuất thư mục `images/` và `manifest.csv` có sẵn cột `content_sha256` chuẩn P-021.

### 2. Tạo Holdout (Val)
* Rút ảnh từ `val/images` (tối đa 1 frame / sequence hành trình).
* Xuất `content_sha256.json` dùng để đăng ký danh sách loại trừ trên P-021 (`POST /api/v2/pools/{id}/holdouts`).
* Xuất kèm ground-truth `labels.json`.

### 3. Gán nhãn batch (Active Learning Loop)
* Nhận file batch từ RoadSift (JSON sample, handoff items, CSV hoặc ZIP).
* Tự động tra cứu ngược SHA-256 ra tên file ảnh gốc BDD100K.
* **Chuẩn hóa nhãn P-021:** Tự động map `person -> pedestrian`, `bike -> bicycle`, `motor -> motorcycle` và lọc bỏ nhãn ngoài taxonomy (tránh lỗi `unknown_class`).
* Xuất file JSON nhãn sẵn sàng nộp vào mục **Nhập annotation** của P-021.

### 4. Xem trước (Preview Ảnh · Metadata · Nhãn)
* Duyệt ảnh từ tập Train, Val hoặc thư mục output đã tạo.
* Render bounding box màu theo từng class trực quan.
* Xem bảng metadata (thời tiết, thời điểm, bối cảnh, sequence, SHA-256) và bảng toạ độ box chi tiết.

---

## 🔄 Tóm tắt luồng làm việc với RoadSift

1. **Chuẩn bị:** Sinh Pool + Holdout trên tool này.
2. **Khởi tạo RoadSift:** Tạo Pool mới trên web RoadSift, đăng ký `content_sha256.json` vào Holdout, rồi upload `images/` + `manifest.csv`.
3. **Vòng lặp Active Learning:**
   * RoadSift chọn batch cần gán nhãn -> bấm **Handoff** tải batch về.
   * Kéo file batch vào **Tab 3** của tool để sinh `bdd100k_batch_labels.json`.
   * Quay lại RoadSift bấm **Nhập annotation** để nộp nhãn và bắt đầu vòng lặp tiếp theo.
