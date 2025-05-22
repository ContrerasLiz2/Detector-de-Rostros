# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:04:00 2025

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
        uic.loadUi("Rostros.ui", self)

    
        self.mp_detectar_rostro = mp.solutions.face_detection
        self.dibujar_rostro = mp.solutions.drawing_utils
        self.detectar_caras = self.mp_detectar_rostro.FaceDetection(min_detection_confidence=0.5)

      
        self.Camaralabel = self.findChild(QtWidgets.QLabel, 'Camaralabel') 
        self.boton_inicio = self.findChild(QtWidgets.QPushButton, 'btninicio')  
   
        self.boton_inicio.clicked.connect(self.iniciar_camara)

   
        self.camara = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)

    def iniciar_camara(self):
        if self.camara is None:
            self.camara = cv2.VideoCapture(0)
            if not self.camara.isOpened():
                print("No se pudo abrir la cámara.")
                return
            self.timer.start(30)  

    def actualizar_frame(self):
        if self.camara is not None:
            ret, frame = self.camara.read()
            if ret:
                frame = cv2.flip(frame, 1) 
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
                resultado = self.detectar_caras.process(rgb)
                if resultado.detections:
                    for deteccion in resultado.detections:
                        self.dibujar_rostro.draw_detection(frame, deteccion)

          
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(self.Camaralabel.width(), self.Camaralabel.height())

               
                self.Camaralabel.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
      
        if self.camara:
            self.camara.release()
        self.timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())

