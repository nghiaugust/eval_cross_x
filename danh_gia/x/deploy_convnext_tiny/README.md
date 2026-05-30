# Deploy ConvNeXt-Tiny / ConvNeXt-Tiny + SVM

Thu muc nay la goi inference cho model ConvNeXt-Tiny da train tren dataset 3 lop:

- `0`: `no_x`
- `1`: `x_cancel`
- `2`: `x_mark`

Preprocess mac dinh dung `input_size: [640, 640]`, dong bo voi cau hinh `config_convnext_tiny.yaml`.

Goi nay ho tro 2 che do:

- `cnn`: nap `best_cnn.pt` va phan loai truc tiep bang classifier cuoi cua ConvNeXt-Tiny.
- `svm`: nap `best_cnn.pt`, trich feature 768 chieu tu ConvNeXt-Tiny, sau do phan loai bang `svm_model.joblib`.

## Dat file trong so

Sau khi train xong, copy weights vao:

```text
deploy_convnext_tiny/
  config.yaml
  predict.py
  weights/
    best_cnn.pt
    svm_model.joblib
```

Neu dang o thu muc `train_model/`:

```powershell
Copy-Item runs\cnn_convnext_tiny\best_cnn.pt deploy_convnext_tiny\weights\best_cnn.pt
Copy-Item runs\svm_convnext_tiny\svm_model.joblib deploy_convnext_tiny\weights\svm_model.joblib
```

Ban cung co the truyen truc tiep duong dan bang `--cnn-checkpoint` va `--svm-model`.

## Cai thu vien

```powershell
pip install -r requirements.txt
```

## Du doan bang CNN thuan

Mot anh:

```powershell
python predict.py --mode cnn --input C:\path\to\image.jpg
```

Ca thu muc anh va luu CSV:

```powershell
python predict.py --mode cnn --input C:\path\to\images --output cnn_predictions.csv
```

## Du doan bang ConvNeXt-Tiny + SVM

Mot anh:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg
```

Ca thu muc anh va luu CSV:

```powershell
python predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Truyen truc tiep ca checkpoint CNN va model SVM:

```powershell
python predict.py --mode svm --input C:\path\to\image.jpg --cnn-checkpoint C:\path\to\best_cnn.pt --svm-model C:\path\to\svm_model.joblib
```

## Ket qua dau ra

CSV hoac JSON output gom:

- `path`: duong dan anh dau vao
- `pred_label`: nhan dang so
- `pred_name`: ten lop du doan
- `prob_no_x`, `prob_x_cancel`, `prob_x_mark`: xac suat tung lop neu model co ho tro

