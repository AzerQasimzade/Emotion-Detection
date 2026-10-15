import cv2

print("OpenCV:", cv2.__version__)

for i in range(5):
    cap = cv2.VideoCapture(i)

    print(f"Camera {i}: opened={cap.isOpened()}")

    if cap.isOpened():
        ret, frame = cap.read()
        print("Frame received:", ret)

    cap.release()