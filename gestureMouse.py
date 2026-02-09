import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

# -------- Volume control ----------
from pycaw.pycaw import AudioUtilities

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()

# -------- MediaPipe ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -------- Screen ----------
screen_w, screen_h = pyautogui.size()

# smoothing
plocX, plocY = 0, 0
smooth = 5

cap = cv2.VideoCapture(0)

# cooldown timers
last_click = 0
cooldown = 0.4

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = []
        for id, landmark in enumerate(result.multi_hand_landmarks[0].landmark):
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            lm.append((cx, cy))

        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        thumb = lm[4]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]
        pinky = lm[20]

        now = time.time()

        # =================================================
        # 🖱 1️⃣ Cursor Move (index only)
        # =================================================
        x = np.interp(index[0], [100, w-100], [0, screen_w])
        y = np.interp(index[1], [100, h-100], [0, screen_h])

        clocX = plocX + (x-plocX)/smooth
        clocY = plocY + (y-plocY)/smooth

        pyautogui.moveTo(clocX, clocY)
        plocX, plocY = clocX, clocY

        # =================================================
        # 👆 Left Click (thumb + index)
        # =================================================
        if distance(thumb, index) < 30 and now-last_click > cooldown:
            pyautogui.click()
            last_click = now

        # =================================================
        # 👉 Right Click (thumb + middle)
        # =================================================
        if distance(thumb, middle) < 30 and now-last_click > cooldown:
            pyautogui.rightClick()
            last_click = now

        # =================================================
        # ✊ Drag (fist)
        # =================================================
        # if distance(index, middle) < 25 and distance(middle, ring) < 25:
        #     pyautogui.mouseDown()
        # else:
        #     pyautogui.mouseUp()

        # =================================================
        # 🖐 Scroll (two fingers up)
        # =================================================
        if abs(index[1]-middle[1]) < 20:
            if index[1] < h//2:
                pyautogui.scroll(40)
            else:
                pyautogui.scroll(-40)

        # =================================================
        # 🔊 Volume (thumb-index distance)
        # =================================================
        # length = distance(thumb, index)
        # vol = np.interp(length, [30, 200], [minVol, maxVol])
        # volume.SetMasterVolumeLevel(vol, None)

        # =================================================
        # 📸 Screenshot (✌ victory)
        # =================================================
        # if (index[1] < middle[1]) and (ring[1] > middle[1]) and (pinky[1] > middle[1]):
        #     pyautogui.hotkey("win", "shift", "s")

        # =================================================
        # 🔇 Mute (open palm)
        # =================================================
        # if thumb[1] < index[1] < middle[1] < ring[1] < pinky[1]:
        #     pyautogui.press("volumemute")

    cv2.imshow("Air Mouse AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
