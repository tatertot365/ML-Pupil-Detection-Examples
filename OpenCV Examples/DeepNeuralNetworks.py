import cv2
import numpy as np
# Load the image and prepare as input
image = cv2.imread('/Users/tategillespie/Desktop/Personal Coding Projects/Face and Eye Detection/Test Files/Images/black-male-1.jpg')
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (104.0, 177.0, 123.0))
# Load a pre-trained deep learning model for face detection
# net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Perform detection
net.setInput(blob)
detections = net.forward()
# Post-process to find eyes within the face region
# Assume eyes network available as 'eye_net'
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = image[startY:endY, startX:endX]
        blob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), (104.0, 177.0, 123.0))
        eye_net.setInput(blob)
        eye_detections = eye_net.forward()
        for j in range(eye_detections.shape[2]):
            confidence = eye_detections[0, 0, j, 2]
            if confidence > 0.5:
                box = eye_detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('Detected Eyes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()