import cv2
import numpy as np
import time

prototxt_path = "weights/deploy.prototxt.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)
while True:
    start = time.time()
    _, image = cap.read()
    h, w = image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if confidence > 0.4:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(np.int)
            face = image[start_y: end_y, start_x: end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            image[start_y: end_y, start_x: end_x] = face
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
    time_elapsed = time.time() - start
    fps = 1 / time_elapsed
    print("FPS:", fps)

cv2.destroyAllWindows()
cap.release()
