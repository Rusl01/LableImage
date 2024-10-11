import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    original_img = img.copy()
    img = cv2.resize(img, input_size) 
    img = img.transpose(2, 0, 1).astype('float32')  
    img /= 255.0 
    img = np.expand_dims(img, axis=0)  
    return img, original_img

def postprocess(outputs, original_img, conf_threshold=0.25, input_size=(640, 640)):
    
    boxes_and_scores = outputs[0].squeeze()  
    
    boxes = boxes_and_scores[:4, :]  
    scores = boxes_and_scores[4, :]  

    scale_x = original_img.shape[1] / input_size[0]
    scale_y = original_img.shape[0] / input_size[1]

    boxes = boxes.transpose()

    for box, score in zip(boxes, scores):
        if score > conf_threshold:  
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return original_img

def show_image(image):
    cv2.imshow("Detected objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ort_session = ort.InferenceSession("D:\\LableImage\\best.onnx")

img, original_img = preprocess_image("D:\\LableImage\\data\\images\\train\\id_170_value_72_507.jpg")

outputs = ort_session.run(None, {"images": img})

for idx, output in enumerate(outputs):
    print(f"Output {idx} shape: {output.shape}")

processed_img = postprocess(outputs, original_img)

show_image(processed_img)
