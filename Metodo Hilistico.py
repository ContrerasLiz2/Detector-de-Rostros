# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:04:42 2025

@author: lizet
"""
import sys
import cv2
import mediapipe as mp
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class MiVentana(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("RostroAnimacion.ui", self)
        
       
        self.mp_holistico = mp.solutions.holistic
        self.dibujar = mp.solutions.drawing_utils
        self.estilo_dibujo = mp.solutions.drawing_styles
        
        
        self.holistico = self.mp_holistico.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
      
        self.camara = cv2.VideoCapture(0)
        
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)  # Actualizar cada 30 ms
        
       
        self.label_video = self.findChild(QtWidgets.QLabel, "label_video")
        
    def actualizar_frame(self):
        ret, frame = self.camara.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
       
        resultado = self.holistico.process(rgb)
        
       
        if resultado.face_landmarks:
            self.dibujar.draw_landmarks(
                frame,
                resultado.face_landmarks,
                self.mp_holistico.FACEMESH_CONTOURS,
                self.dibujar.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )
        
        
        if resultado.pose_landmarks:
            self.dibujar.draw_landmarks(
                frame,
                resultado.pose_landmarks,
                self.mp_holistico.POSE_CONNECTIONS,
                self.dibujar.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
     
        if resultado.left_hand_landmarks:  # Corregido el nombre (era left_hand_lanmarks)
            self.dibujar.draw_landmarks(
                frame,
                resultado.left_hand_landmarks,
                self.mp_holistico.HAND_CONNECTIONS,
                self.dibujar.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        
      
        if resultado.right_hand_landmarks:  
            self.dibujar.draw_landmarks(
                frame,
                resultado.right_hand_landmarks,
                self.mp_holistico.HAND_CONNECTIONS,
                self.dibujar.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )
        
       
        h, w, ch = frame.shape
        bytes_per_line = ch * w
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convert_to_Qt_format = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.label_video.width(), 
            self.label_video.height()
        )
        
      
        self.label_video.setPixmap(QPixmap.fromImage(p))
    
    def closeEvent(self, event):
       
        self.camara.release()
        self.holistico.close()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())