import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    original_img = img.copy()
    img = cv2.resize(img, input_size)  # Resize image
    img = img.transpose(2, 0, 1).astype('float32')  # Convert to (3, H, W) format
    img /= 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, original_img

def postprocess(outputs, original_img, conf_threshold=0.25, input_size=(640, 640)):
    # Assuming output is (1, 5, 8400) - 1 batch, 5 values (x1, y1, x2, y2, score), 8400 detections
    boxes_and_scores = outputs[0].squeeze()  # Remove the batch dimension: shape becomes (5, 8400)
    
    # Separate boxes and scores
    boxes = boxes_and_scores[:4, :]  # First 4 rows are x1, y1, x2, y2
    scores = boxes_and_scores[4, :]  # Last row is the scores

    scale_x = original_img.shape[1] / input_size[0]
    scale_y = original_img.shape[0] / input_size[1]

    # Transpose boxes to make them in the form (8400, 4) so we can iterate easily
    boxes = boxes.transpose()

    # Iterate over boxes and scores
    for box, score in zip(boxes, scores):
        if score > conf_threshold:  # Apply confidence threshold
            x1, y1, x2, y2 = box
            # Scale box coordinates back to the original image size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Draw the bounding box on the original image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return original_img

def show_image(image):
    cv2.imshow("Detected objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the ONNX model
ort_session = ort.InferenceSession("D:\\LableImage\\best.onnx")

# Preprocess the image
img, original_img = preprocess_image("D:\\LableImage\\data\\images\\train\\id_170_value_72_507.jpg")

# Perform inference
outputs = ort_session.run(None, {"images": img})

# Print the output shapes for debugging
for idx, output in enumerate(outputs):
    print(f"Output {idx} shape: {output.shape}")

# Process the results
processed_img = postprocess(outputs, original_img)

# Show the result
show_image(processed_img)
