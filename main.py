import cv2
from hand_detector import HandDetector

detector = HandDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame, fingers = detector.detect_hands(frame)

    cv2.putText(
        frame,
        f"Fingers: {fingers}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()