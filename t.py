import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1)
mpDraw = mp.solutions.drawing_utils 
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Не удалось получить кадр с web-камеры")
      continue
    image = cv2.flip(image, 1)
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB_image)
    multiLandMarks = results.multi_hand_landmarks
    
    if multiLandMarks:
        mLM = list(enumerate(multiLandMarks))
        for idx, handLms in enumerate(multiLandMarks):
            #print(multiLandMarks[idx].classification[0])
            print(results.multi_handedness[idx].classification)
            lbl = results.multi_handedness[idx].classification[0].label
            print(lbl)
        upCount = 0
        for handLms in multiLandMarks:
            handList = []
            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))
            for point in handList:
                cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)

            for coordinate in finger_Coord:
                if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                    upCount += 1
            if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
                upCount += 1

        cv2.putText(image, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
       
    cv2.imshow("Counting number of fingers", image)
    if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()