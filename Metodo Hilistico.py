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

        self.Camaralabel = self.findChild(QtWidgets.QLabel, "Camaralabel")
        self.btnCara = self.findChild(QtWidgets.QPushButton, "btnCara")
        self.btnCuerpo = self.findChild(QtWidgets.QPushButton, "btnCuerpo")

        self.btnCara.clicked.connect(self.activar_cara)
        self.btnCuerpo.clicked.connect(self.activar_cuerpo)

        self.camara = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.modo = "ninguno"

        self.mp_dibujo = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_mesh
        self.mp_holistico = mp.solutions.holistic

        self.face_mesh = self.mp_face.FaceMesh()
        self.holistico = self.mp_holistico.Holistic()

    def activar_cara(self):
        self.modo = "cara"
        self.timer.start(30)

    def activar_cuerpo(self):
        self.modo = "cuerpo"
        self.timer.start(30)

    def actualizar_frame(self):
        ret, frame = self.camara.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.modo == "cara":
            resultado = self.face_mesh.process(rgb)
            if resultado.multi_face_landmarks:
                for rostro in resultado.multi_face_landmarks:
                    self.mp_dibujo.draw_landmarks(
                        frame, rostro, self.mp_face.FACEMESH_TESSELATION)

        elif self.modo == "cuerpo":
            resultado = self.holistico.process(rgb)
            if resultado.face_landmarks:
                self.mp_dibujo.draw_landmarks(
                    frame, resultado.face_landmarks, self.mp_holistico.FACEMESH_TESSELATION)
            if resultado.pose_landmarks:
                self.mp_dibujo.draw_landmarks(
                    frame, resultado.pose_landmarks, self.mp_holistico.POSE_CONNECTIONS)

        img = QImage(frame.data, frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format_BGR888)
        self.Camaralabel.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.camara.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())