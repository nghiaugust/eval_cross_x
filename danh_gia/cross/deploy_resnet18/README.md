# Deploy ResNet18 / ResNet18 + SVM

Thư mục này là gói chạy thật cho cấu hình ResNet18.

Gói này hỗ trợ 2 chế độ dự đoán:

- `cnn`: nạp `best_cnn.pt` và phân loại trực tiếp bằng lớp FC cuối của CNN.
- `svm`: nạp `best_cnn.pt`, trích xuất vector đặc trưng 512 chiều từ CNN, sau đó phân loại bằng `svm_model.joblib`.

## 1. Đặt file trọng số

Cấu trúc khuyến nghị:

```text
deploy_resnet18/
  config.yaml
  predict.py
  weights/
    best_cnn.pt
    svm_model.joblib
```

Sau khi huấn luyện xong, copy file trọng số vào thư mục `weights/`:

```powershell
Copy-Item ..\runs\cnn_resnet18\best_cnn.pt .\weights\best_cnn.pt
Copy-Item ..\runs\svm_resnet18\svm_model.joblib .\weights\svm_model.joblib
```

Nếu đang chạy lệnh từ thư mục gốc `train_model/`, dùng:

```powershell
Copy-Item runs\cnn_resnet18\best_cnn.pt deploy_resnet18\weights\best_cnn.pt
Copy-Item runs\svm_resnet18\svm_model.joblib deploy_resnet18\weights\svm_model.joblib
```

Đường dẫn mặc định của file trọng số nằm trong `config.yaml`:

```yaml
paths:
  cnn_checkpoint: weights/best_cnn.pt
  svm_model: weights/svm_model.joblib
```

Bạn có thể không cần copy file vào `weights/`; khi chạy dự đoán, truyền trực tiếp đường dẫn bằng `--cnn-checkpoint` và `--svm-model`.

## 2. Cài thư viện cần thiết

```powershell
pip install -r requirements.txt
```

Nếu đã cài `requirements.txt` của project chính thì thường không cần cài lại bước này.

## 3. Dự đoán bằng CNN thuần

Dự đoán một ảnh:

```powershell
python predict.py --mode cnn --input C:\path\to\image.jpg
```

Dự đoán cả thư mục ảnh và lưu kết quả ra CSV:

```powershell
python predict.py --mode cnn --input C:\path\to\images --output cnn_predictions.csv
```

Truyền trực tiếp đường dẫn checkpoint CNN:

```powershell
python predict.py --mode cnn --input C:\path\to\image.jpg --cnn-checkpoint C:\path\to\best_cnn.pt
```

## 4. Dự đoán bằng ResNet18 + SVM

Dự đoán một ảnh:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg
```

Dự đoán cả thư mục ảnh và lưu kết quả ra CSV:

```powershell
python predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Truyền trực tiếp cả checkpoint CNN và model SVM:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg --cnn-checkpoint C:\path\to\best_cnn.pt --svm-model C:\path\to\svm_model.joblib
```

## 5. Kết quả đầu ra

Kết quả dự đoán gồm các cột:

- `path`: đường dẫn ảnh đầu vào.
- `pred_label`: nhãn dạng số.
- `pred_name`: tên lớp dự đoán, gồm `Gach_Ten` hoặc `Ten`.
- `prob_Gach_Ten`, `prob_Ten`: xác suất từng lớp nếu mô hình có hỗ trợ.
