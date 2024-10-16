import cv2
import numpy as np
import onnxruntime as ort

# Загрузка модели ONNX
# session = ort.InferenceSession("D:\\LableImage\\runs\\detect\\train5\\weights\\best.onnx") 
session = ort.InferenceSession("D:\\LableImage\\best.onnx") 
# Загрузка изображения
image = cv2.imread('D:\\LableImage\\data\\images\\train\\id_170_value_72_507.jpg')  
image_resized = cv2.resize(image, (640, 640))
image_np = np.transpose(image_resized, (2, 0, 1)).astype(np.float32)
image_np = np.expand_dims(image_np, axis=0)

# Предсказание
outputs = session.run(None, {session.get_inputs()[0].name: image_np})

# Обработка выводов

pred_boxes = outputs[0][0]  
boxes = pred_boxes[:, :4]  
confidences = pred_boxes[:, 4]  # Уверенности
class_ids = pred_boxes[:, 5].astype(int)  # Идентификаторы классов

# Фильтрация предсказаний по уверенности
threshold = 0.6 
for box, confidence, class_id in zip(boxes, confidences, class_ids):
    if confidence >= threshold:
        x1, y1, x2, y2 = box.astype(int)
        
        # Рисуем прямоугольник на изображении
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f'Class: {class_id}, Conf: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Отображаем изображение
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()