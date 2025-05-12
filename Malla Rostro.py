# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:35:38 2025

@author: lizet
"""


import cv2
import mediapipe as mp

mp_malla_cara = mp.solutions.face_mesh
malla = mp_malla_cara.FaceMesh(max_num_faces=2, refine_landmarks=True)

mp_dibujo = mp.solutions.drawing_utils
dibujo_especificaciones = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)

camara = cv2.VideoCapture(0)

while camara.isOpened():
    r, frame = camara.read()
    
    if not r:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resultado = malla.process(rgb)
    
    if resultado.multi_face_landmarks:
        for puntos_rostro in resultado.multi_face_landmarks:
            mp_dibujo.draw_landmarks(
                image=frame,
                landmark_list=puntos_rostro,
                connections=mp_malla_cara.FACEMESH_TESSELATION,
                landmark_drawing_spec=dibujo_especificaciones,
                connection_drawing_spec=dibujo_especificaciones
            )
    
    cv2.imshow("Rostro", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
