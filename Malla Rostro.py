# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:35:38 2025

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
        uic.loadUi("MascaraRostro.ui", self)
        
        self.mp_malla_cara = mp.solutions.face_mesh
        self.malla = self.mp_malla_cara.FaceMesh(max_num_faces=2, refine_landmarks=True)
        
        self.mp_dibujo = mp.solutions.drawing_utils
        self.dibujo_especificaciones = self.mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)
        
        self.camara = cv2.VideoCapture(0)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30) 
        
        self.label_video = self.findChild(QtWidgets.QLabel, "label_video")
        
    def actualizar_frame(self):
        ret, frame = self.camara.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resultado = self.malla.process(rgb)
        
        if resultado.multi_face_landmarks:
            for puntos_rostro in resultado.multi_face_landmarks:
                self.mp_dibujo.draw_landmarks(
                    image=frame,
                    landmark_list=puntos_rostro,
                    connections=self.mp_malla_cara.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.dibujo_especificaciones,
                    connection_drawing_spec=self.dibujo_especificaciones
                )
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label_video.width(), self.label_video.height())
        
        self.label_video.setPixmap(QPixmap.fromImage(p))
    
    def closeEvent(self, event):
        self.camara.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())