import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

        # Thumb, Index, Middle, Ring, Pinky
        self.tip_ids = [4, 8, 12, 16, 20]

    def detect_hands(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        total_fingers = 0

        if results.multi_hand_landmarks:

            h, w, _ = frame.shape

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = []

                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks.append((x, y))

                finger_count = 0

                # Thumb
                if landmarks[4][0] > landmarks[3][0]:
                    finger_count += 1

                # Index, Middle, Ring, Pinky
                for tip in [8, 12, 16, 20]:

                    if landmarks[tip][1] < landmarks[tip - 2][1]:
                        finger_count += 1

                total_fingers += finger_count

        return frame, total_fingers