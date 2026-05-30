# Deploy ResNet50 / ResNet50 + SVM

Thu muc nay chay inference cho model ResNet50 da train bang `config_resnet50.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\cnn_resnet50\best_cnn.pt deploy_resnet50\weights\best_cnn.pt
Copy-Item runs\svm_resnet50\svm_model.joblib deploy_resnet50\weights\svm_model.joblib
```

Hoac truyen duong dan truc tiep bang `--cnn-checkpoint` va `--svm-model`.

## Cai dependencies

```powershell
pip install -r requirements.txt
```

## Chay predict

CNN only:

```powershell
python deploy_resnet50\predict.py --mode cnn --input C:\path\to\image.jpg
```

CNN + SVM:

```powershell
python deploy_resnet50\predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*` neu model co xac suat.
