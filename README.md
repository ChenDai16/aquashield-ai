# AquaShield AI

Công cụ AI sàng lọc sớm sức khỏe tôm tại ao. Từ một tấm ảnh tôm sú (*Penaeus monodon*) chụp bằng điện thoại, mô hình xếp ảnh vào 4 nhóm, chỉ ra vùng ảnh nó dựa vào (Grad-CAM) và nêu việc cần làm trong 24 giờ.

**Dùng thử trên điện thoại:** https://chendai16.github.io/aquashield-ai/app/

Mô hình chạy ngay trong máy, ảnh không rời máy. Sau lần mở đầu tiên có mạng, ứng dụng chạy được ở chế độ máy bay.

| Lớp | Mã | Ý nghĩa |
|---|---|---|
| 0 | BG | Dấu hiệu đen mang |
| 1 | BG_WSSV | Dấu hiệu đen mang kèm đốm trắng |
| 2 | HS | Chưa thấy dấu hiệu bất thường |
| 3 | WSSV | Dấu hiệu đốm trắng |

## Kết quả kiểm định

Mô hình ResNet-18 học chuyển giao từ ImageNet. Trên tập kiểm tra nội bộ 96 ảnh (24 ảnh mỗi lớp), bản PyTorch FP32 đạt:

- đúng lớp 90/96 = 93,75%;
- độ nhạy với dấu hiệu đốm trắng (gộp WSSV và BG_WSSV) 46/48 = 95,8%;
- độ đặc hiệu 47/48 = 97,9%.

Chi tiết từng lớp ở `results/classification_report_sach_n96.txt`. Kết quả của bản INT8 chạy trên điện thoại được chấm riêng trên đúng 96 ảnh này và hiển thị trong tab "Thông tin" của ứng dụng.

**Con số 98,05% đã được rút lại.** Con số đó tính trên thư mục `Test` gốc của bộ dữ liệu. Khi rà soát bằng mã băm MD5, toàn bộ 256 ảnh của thư mục này trùng với ảnh huấn luyện. Nhóm đã loại ảnh trùng, giữ 640 ảnh, chia lại 448/96/96 và huấn luyện lại.

## Giới hạn

- Chỉ dành cho tôm sú.
- Ảnh chủ yếu lấy từ bộ dữ liệu mở ShrimpDiseaseImageBD (Bangladesh), nhãn theo quan sát, chưa đối chiếu PCR.
- Chưa được kiểm chứng trên ảnh chụp tại ao nuôi Việt Nam.
- Bộ dữ liệu ban đầu có cả ảnh tăng cường (xoay, lật, đổi độ sáng). Nếu các biến thể của cùng một ảnh gốc nằm ở cả tập huấn luyện và tập kiểm tra, kết quả nội bộ sẽ cao hơn hiệu năng thực tế.
- Khi độ tin cậy dưới 55%, ứng dụng không kết luận và yêu cầu chụp lại.
- Đây là công cụ sàng lọc. Nó không thay thế xét nghiệm PCR và cán bộ thú y thủy sản, và không dùng để kê thuốc.

## Thư mục

```
app/        Ứng dụng điện thoại chạy ngoại tuyến (PWA): giao diện, thư viện onnxruntime-web, mô hình INT8
model/      Trọng số PyTorch (resnet18_transfer.pt), ONNX FP32, nhãn
mobile/     Bản thử cũ (đã chuyển hướng sang app/) và các tệp ONNX cũ
results/    Báo cáo kiểm định
app.py      Bản chạy trên máy tính (Gradio)
export_onnx.py, evaluate.py   Xuất ONNX và đánh giá
```

## Chi tiết kỹ thuật

- Kiến trúc: ResNet-18, lớp `fc` thay bằng `Linear(512 → 4)`.
- Tiền xử lý: ép ảnh về 224×224, chuẩn hóa theo ImageNet.
- Bản điện thoại: ONNX lượng tử hóa INT8 khoảng 11 MB, chạy bằng onnxruntime-web (WebAssembly).
- Grad-CAM tính trên `layer4`. Với ResNet-18 (gộp trung bình toàn cục rồi một lớp tuyến tính), Grad-CAM bằng đúng tổng có trọng số của đặc trưng lớp cuối theo trọng số lớp `fc`, nên được tính ngay trong mô hình ONNX.

## Dữ liệu

Islam, M. M., Sarker, A., Choudhury, A., Ahmed, N., & Shafi, A. A. (2024). *ShrimpDiseaseImageBD: An image dataset for computer vision-based detection of shrimp diseases in Bangladesh* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/jhrtdj9txm.2 (CC BY 4.0).

---
Trần Quang Đại, ngành Nuôi trồng Thủy sản, Trường Đại học Nha Trang.
