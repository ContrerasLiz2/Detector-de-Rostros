# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:16:29 2025

@author: lizet

"""

import cv2 
from pycaw.pycaw import AudioUtilities
from pycaw.pycaw import IAudioEndpointVolume
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL



dispositivo = AudioUtilities.GetSpeakers()
interfaz = dispositivo.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
control_volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
dibujar = mp.solutions.drawing_utils


mp_dibujo = mp.solutions.drawing_utils
mp_holistico = mp.solutions.holistic
modelo = mp_holistico.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

camara = cv2.VideoCapture(0)

def contar_dedos(mano):
    dedos = []
    for punta in {8,12,16,20}:
        if mano.landmark[punta].y < mano.landmark[punta - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)
            
    if mano.landmark[4].x > mano.landmark[2].x:
        dedos.append(1)
    else:
        dedos.append(0)
    return sum(dedos)

while camara.isOpened():
    r, frame = camara.read()
    
    
    if not r:
       break
   
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    resultado = modelo.process(rgb)
    
    total_dedos = 0
    
    if resultado.right_hand_landmarks:
        mp_dibujo.draw_landmarks(frame, resultado.right_hand_landmarks,mp_holistico.HAND_CONNECTIONS)
        
        
        total_dedos = contar_dedos(resultado.right_hand_landmarks)
        
        volumen_escala = total_dedos / 5
        control_volumen.SetMasterVolumeLevelScalar(volumen_escala, None)
        
        
    cv2.putText(frame, f"Dedos: {total_dedos}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
    
    cv2.imshow("Volumen: ", frame)
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camara.release()
cv2.destroyAllWindows()