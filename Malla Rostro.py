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
        uic.loadUi("MascaradeRostro.ui", self)  

        self.label_video = self.findChild(QtWidgets.QLabel, "label_video")
        self.btnRostroinicio = self.findChild(QtWidgets.QPushButton, "btnRostroinicio")
        self.btnRostro = self.findChild(QtWidgets.QPushButton, "btnRostro")

        self.btnRostroinicio.clicked.connect(self.activar_rostro)
        self.btnRostro.clicked.connect(self.activar_rostro)
        self.camara = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(max_num_faces=2, refine_landmarks=True)

        self.mp_dibujo = mp.solutions.drawing_utils
        self.estilo_dibujo = self.mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)

        self.modo = False  

    def activar_rostro(self):
        self.modo = True
        self.timer.start(30)

    def actualizar_frame(self):
        ret, frame = self.camara.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.modo:
            resultado = self.face_mesh.process(rgb)
            if resultado.multi_face_landmarks:
                for puntos in resultado.multi_face_landmarks:
                    self.mp_dibujo.draw_landmarks(
                        image=frame,
                        landmark_list=puntos,
                        connections=self.mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.estilo_dibujo,
                        connection_drawing_spec=self.estilo_dibujo
                    )

        img = QImage(frame.data, frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format_BGR888)
        self.label_video.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.camara.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())
