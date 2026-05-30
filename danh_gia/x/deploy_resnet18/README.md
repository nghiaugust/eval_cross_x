# Deploy ResNet18 / ResNet18 + SVM

Thư mục này là gói chạy inference cho model ResNet18 đã train trên dataset 3 lớp:

- `0`: `no_x`
- `1`: `x_cancel`
- `2`: `x_mark`

Preprocess mặc định dùng `input_size: [640, 640]`, đồng bộ với cấu hình train hiện tại.

Gói này hỗ trợ 2 chế độ:

- `cnn`: nạp `best_cnn.pt` và phân loại trực tiếp bằng lớp FC cuối của CNN.
- `svm`: nạp `best_cnn.pt`, trích feature 512 chiều từ CNN, sau đó phân loại bằng `svm_model.joblib`.

## Đặt file trọng số

Sau khi train lại, copy weights mới vào:

```text
deploy_resnet18/
  config.yaml
  predict.py
  weights/
    best_cnn.pt
    svm_model.joblib
```

Nếu đang ở thư mục `train_model/`:

```powershell
Copy-Item runs\cnn_resnet18\best_cnn.pt deploy_resnet18\weights\best_cnn.pt
Copy-Item runs\svm_resnet18\svm_model.joblib deploy_resnet18\weights\svm_model.joblib
```

Bạn cũng có thể truyền trực tiếp đường dẫn bằng `--cnn-checkpoint` và `--svm-model`.

## Cài thư viện

```powershell
pip install -r requirements.txt
```

## Dự đoán bằng CNN thuần

Một ảnh:

```powershell
python predict.py --mode cnn --input C:\path\to\image.jpg
```

Cả thư mục ảnh và lưu CSV:

```powershell
python predict.py --mode cnn --input C:\path\to\images --output cnn_predictions.csv
```

## Dự đoán bằng ResNet18 + SVM

Một ảnh:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg
```

Cả thư mục ảnh và lưu CSV:

```powershell
python predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Truyền trực tiếp cả checkpoint CNN và model SVM:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg --cnn-checkpoint C:\path\to\best_cnn.pt --svm-model C:\path\to\svm_model.joblib
```

## Kết quả đầu ra

CSV hoặc JSON output gồm:

- `path`: đường dẫn ảnh đầu vào
- `pred_label`: nhãn dạng số
- `pred_name`: tên lớp dự đoán
- `prob_no_x`, `prob_x_cancel`, `prob_x_mark`: xác suất từng lớp nếu model có hỗ trợ
