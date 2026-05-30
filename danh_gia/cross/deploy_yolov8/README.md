# Deploy YOLOv8 Classification

Thu muc nay chay inference cho YOLOv8 classification da train bang `config_yolov8.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\classify\runs\yolov8_cls\yolov8n_cls\weights\best.pt deploy_yolov8\weights\best.pt
```

Hoac truyen duong dan truc tiep:

```powershell
python deploy_yolov8\predict.py --input C:\path\to\image.jpg --checkpoint runs\yolov8_cls\yolov8n_cls\weights\best.pt
```

## Cai dependencies

```powershell
pip install -r requirements.txt
```

## Chay predict

Mot anh:

```powershell
python deploy_yolov8\predict.py --input C:\path\to\image.jpg
```

Ca thu muc anh va luu CSV:

```powershell
python deploy_yolov8\predict.py --input C:\path\to\images --output yolov8_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*`.
