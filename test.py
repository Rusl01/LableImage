import onnxruntime as ort
from PIL import Image, ImageDraw
import numpy as np

# Загрузка модели ONNX
model_path = 'runs/detect/train5/weights/best.onnx'
session = ort.InferenceSession(model_path)

# Функция для предобработки изображения
def preprocess_image(image_path, img_size=640):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((img_size, img_size))  # Изменение размера изображения
    image_data = np.asarray(image_resized, dtype=np.float32) / 255.0  # Нормализация
    image_data = np.transpose(image_data, (2, 0, 1))  # Изменение порядка осей (C, H, W)
    image_data = np.expand_dims(image_data, axis=0)  # Добавление размерности для батча
    return image, image_data

# Функция для отрисовки рамок на изображении
def draw_boxes(image, predictions, img_size=640, conf_threshold=0.5):
    draw = ImageDraw.Draw(image)
    
    for pred in predictions:
        confidence = pred[4]
        if confidence < conf_threshold:
            continue

        x_center, y_center, width, height = pred[:4]
        
        x1 = int((x_center - width / 2) / img_size * image.size[0])
        y1 = int((y_center - height / 2) / img_size * image.size[1])
        x2 = int((x_center + width / 2) / img_size * image.size[0])
        y2 = int((y_center + height / 2) / img_size * image.size[1])

        # Отрисовка рамки
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        draw.text((x1, y1), f"{confidence:.2f}", fill='red')

# Путь к тестовому изображению
image_path = 'D:\\LableImage\\data\\images\\train\\id_194_value_153_434.jpg'
original_image, input_image = preprocess_image(image_path)

# Выполнение предсказания
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predictions = session.run([output_name], {input_name: input_image})[0]

# Структура выходных данных predictions:
# [x_center, y_center, width, height, confidence] для каждого объекта
print(predictions.shape)  # Выведем размерность массива
print(predictions[0][:5])  # Выведем первые пять значений предсказаний

# Визуализация предсказаний на изображении
draw_boxes(original_image, predictions[0], img_size=640)

# Сохранение и отображение изображения
output_image_path = "output_test_image_with_boxes.jpg"
original_image.save(output_image_path)
original_image.show()

print(f"Результат сохранён как {output_image_path}")
