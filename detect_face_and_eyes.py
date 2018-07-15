import cv2
import os.path as path

dirPath = path.dirname(path.abspath(__file__))
cascadePath = dirPath + '/haarcascade_files/'

face_cascade = cv2.CascadeClassifier(cascadePath + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascadePath + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color, (ex+ew/2, ey+eh/2), min(ew/2, eh/2), (0, 255, 0), 2)

    cv2.imshow("video", frame)
    key = cv2.waitKey(40)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
