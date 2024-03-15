import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Process the image
image = cv2.imread('/Users/tategillespie/Desktop/Personal Coding Projects/Face and Eye Detection/Test Files/Images/black-male-1.jpg')
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Assuming faces are detected, extract the landmarks for the eyes
# and draw them (Omitted for simplicity)

cv2.imshow('Detected Eyes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()