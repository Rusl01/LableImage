from ultralytics import YOLO

# Загрузка модели YOLOv8n
model = YOLO('yolov8n.pt')

# Обучение модели на твоих данных
model.train(data='D:\\LableImage\\data\\data.yaml', epochs=50, imgsz=640)

model.export(format='onnx')