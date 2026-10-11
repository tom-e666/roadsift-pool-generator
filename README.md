# BDD100K Pool Tool

Ứng dụng Streamlit chạy cục bộ, nằm ngoài repo RoadSift.

## Chạy ứng dụng

Mở `run.bat`. Lần đầu, Conda tạo môi trường trong `.conda-env` rồi mở giao diện trên trình duyệt.

## Các chức năng

- **Tạo Pool:** rút ngẫu nhiên ảnh JPG từ `train/images`, không kèm nhãn.
- **Tạo Holdout:** lấy ảnh từ `val/images`, tối đa một ảnh mỗi chuỗi; chuỗi đã dùng ở Pool bị loại. Nhãn ground truth được lưu riêng trong `holdout/labels.json`.
- **Nhập batch · Xuất nhãn:** dán đường dẫn tới thư mục/ZIP batch, CSV manifest hoặc JSON lấy từ `GET /api/v2/batches/{batch_id}/samples`. JSON RoadSift được đọc theo `sample.id`, `sample.attributes.source_name` và tự nhận `source_split`; CSV cần cột `source_name,sample_id` (có thể thêm `source_split`). File xuất là BDD100K JSON nhưng `name` dùng sample ID RoadSift, để có thể đưa vào RoadSift > Nhập annotation. Các lớp `pedestrian`, `bicycle`, `motorcycle` được chuẩn hóa thành `person`, `bike`, `motor` theo taxonomy RoadSift.

Đường dẫn archive và output chỉnh ở đầu giao diện. `manifest.csv` của Pool/Holdout lưu index, ID và tên file nguồn; seed giúp tái tạo lượt chọn. Công cụ không gọi API hoặc ghi dữ liệu vào RoadSift: sau khi tải JSON, vào RoadSift > Nhập annotation, chọn batch tương ứng và định dạng BDD100K JSON. Batch phải ở trạng thái `HANDED_OFF` để RoadSift nhận nhãn; backend sẽ kiểm tra membership, taxonomy và annotation trước khi ghi nhận.
