import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class GestureVolumeControl:


    def __init__(self, mode=False, maxHands=1, complexity=1,detectionCon=0.5, trackCon=0.5) -> None:
        self._mode = mode
        self._maxHands = maxHands
        self._complexity = complexity
        self._detectionCon = detectionCon
        self._trackCon = trackCon

        cv.namedWindow("Gesture Volume Control")

        self._cam = cv.VideoCapture(0)
        self._cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self._cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self._cam.set(cv.CAP_PROP_FPS, 5)

        self.mpHands = mp.solutions.hands

        """
        Parâmetros:
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1, (0 ou 1)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5

        Obs: A classe utiliza somente cores RGB. Portanto é necessário converter
        a imagem para RGB.
        """
        self.hands = self.mpHands.Hands(self._mode, self._maxHands, self._complexity, self._detectionCon, self._trackCon)

        # Serve para desenhar os 21 pontos da mão a ser detectada
        self.mp_draw = mp.solutions.drawing_utils

        self.audio_device = AudioUtilities.GetSpeakers()
        self.audio_interface = self.audio_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.audio_interface, POINTER(IAudioEndpointVolume))

        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]

        self.vol_percent = 0

        self.prev_time = 0
        self.curr_time = 0
        
       
    def _findHands(self, frame) -> None:

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
 
        # Processar a imagem RGB
        results = self.hands.process(frame_rgb)

        finger_pos = []
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                for id, lm in enumerate(hlm.landmark):
                    height, width, channel = frame.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    
                    if id == 4 or id == 8:
                        finger_pos.append((cx, cy))
                        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)

                        if len(finger_pos) >= 2: frame = self._change_volume(frame, finger_pos)
                                                  
                self.mp_draw.draw_landmarks(frame, hlm, self.mpHands.HAND_CONNECTIONS)
        
        finger_pos.clear()

        return frame


    def _change_volume(self, frame, finger_coord: list):
        x1, y1 = finger_coord[0]
        x2, y2 = finger_coord[1]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        length = math.hypot(x2 - x1, y2 - y1)
       
        vol = np.interp(length, [50, 300], [self.min_vol, self.max_vol])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        self.vol_percent = np.interp(length, [50, 300], [0, 100])
                
        frame = cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        if length < 50:
            frame = cv.circle(frame, (cx, cy), 8, (255, 0, 255), cv.FILLED)
            
        frame = cv.rectangle(frame, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv.FILLED)
                       
        self.volume.SetMasterVolumeLevel(vol, None)

        return frame
    

    def _showFps(self, frame):
        self.curr_time = time.time()
        fps = int(1 / (self.curr_time - self.prev_time))

        self.prev_time = self.curr_time
        
        return cv.putText(frame, f"fps: {fps}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
                   
           
    def run(self):
        
        while self._cam.isOpened():
            success, frame = self._cam.read()

            frame = self._showFps(frame)

            frame = cv.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
            frame = cv.putText(frame, f"Vol: {int(self.vol_percent)}", (50, 450), cv.FONT_HERSHEY_PLAIN, 
                                2, (255, 0, 0), 2)
            
            frame = self._findHands(frame)
                                                            
            cv.imshow('Gesture Volume Control', frame)
                        
            if cv.waitKey(2) == ord('q'):
                break