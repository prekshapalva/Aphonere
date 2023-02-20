import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

""" Initialization of mediapipe object
def __init__ (self,
              static_image_mode=False,
              max_num_hands=2,
              min_detection_confidence=0.5,
              min_tracking_confidence=0.5) """

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    results = hands.process(img)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((255,0,0), 5, 2),
                                   mp_draw.DrawingSpec((0,255,0), 2, 2))

    cv2.imshow("Video", img)

    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
