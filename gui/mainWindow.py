import os
import sys
import time
import pickle
import psutil

import numpy as np

import mediapipe as mp
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget, QLineEdit, QSpinBox)

# Parâmetros que auxiliam na ilustração/desenho dos landmarks (ponto de referência) das mãos detectadas na nossa imagem
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar o MediaPipe
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.3, max_num_hands=1)


class Thread(QThread):
    updateFrame = Signal(QImage)
    cropFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True
        self.min_porcent = 0.999
        self.prediction_class = ''
        self.prediction_porcent = ''

    def writeText(self, img, text, color=(0, 0, 255)):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        text = text.upper()
        cv2.putText(img, text, (10, 25), fonte, 1.0, color, 1, cv2.LINE_AA)
        cv2.putText(img, text, (11, 26), fonte, 1.0, color, 1, cv2.LINE_AA)

    def cropHand(self, img, cord1, cord2):
        img_copy = img.copy()
        offset = 20
        min_x = int(cord1[0]) - offset
        min_y = int(cord1[1]) - offset
        max_x = int(cord2[0]) + offset
        max_y = int(cord2[1]) + offset

        min_x = 0 if min_x < 0 else min_x
        min_y = 0 if min_y < 0 else min_y
        max_x = 0 if max_x < 0 else max_x
        max_y = 0 if max_y < 0 else max_y

        cropped_img = img_copy[min_y:max_y, min_x:max_x]
        #cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (255,0,0), 5)
        #cv2.putText(img_copy, f'{class_pred[0]} -- {(class_porcent*100):.2f}%', (min_x,min_y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,0), 1, cv2.LINE_AA)
        return cropped_img

    def getCPU(self):
        return psutil.cpu_percent(interval=1)

    def getRAM(self):
        return psutil.virtual_memory()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        labels_dict = {'0': '0', '1': '1', '2': '2', '3': '3',
                       '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
        model_dict = pickle.load(open('./models/MPLClassifier.p', 'rb'))
        model = model_dict['model_mlp']

        while True:
            data_aux = []
            aux_cord_pixel = []
            cord_pixel = []

            model_dict = pickle.load(open('./models/MPLClassifier.p', 'rb'))
            model = model_dict['model_mlp']

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_img = cv2.resize(frame_rgb, (155, 155))
            cropped_img_copy = np.copy(cropped_img)

            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Desenha os landmarks (pontos de referência) sobre as mãos da imagem, identificando a mesma e cada dedo - útil para saber onde estão os landmarks
                    mp_drawing.draw_landmarks(
                        frame_rgb,  # Image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connection
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in result.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        if len(hand_landmarks.landmark) <= 42:
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            aux_cord_pixel.append(x*w)
                            aux_cord_pixel.append(y*h)

                            cord_pixel.append(aux_cord_pixel)

                            data_aux.append(x)
                            data_aux.append(y)
                            aux_cord_pixel = []

                # print(cord_pixel)

                min_point_x = min(cord_pixel, key=lambda pnt: pnt[0])
                min_point_y = min(cord_pixel, key=lambda pnt: pnt[1])
                max_point_x = max(cord_pixel, key=lambda pnt: pnt[0])
                max_point_y = max(cord_pixel, key=lambda pnt: pnt[1])
                min_points = []
                max_points = []

                min_points.append(min_point_x[0])
                min_points.append(min_point_y[1])

                max_points.append(max_point_x[0])
                max_points.append(max_point_y[1])

                prediction = model.predict(np.asarray([data_aux]))
                proba_porcent = model.predict_proba(np.asarray([data_aux]))
                prediction_porcent = proba_porcent.max()

                cropped_img = self.cropHand(frame_rgb, min_points, max_points)
                cropped_img = cv2.resize(cropped_img, (155, 155))
                cropped_img_copy = np.copy(cropped_img)

                print(prediction[0])
                print(f'Predições porcentagem: {prediction_porcent}')
                #predicted_character = labels_dict[prediction[0]]

                self.prediction_class = str(prediction[0])
                self.prediction_porcent = f'{(prediction_porcent*100):.2f}%'

                # print(predicted_character)
                if prediction_porcent.max() >= self.min_porcent:
                    self.writeText(
                        frame_rgb, f'Number: {prediction[0]} - {(prediction_porcent*100):.2f}%')
                else:
                    self.writeText(frame_rgb, 'Low porcent')
            else:
                self.writeText(frame_rgb, 'Number: NaN', (255, 0, 0))

            # Creating and scaling QImage
            h, w, ch = frame_rgb.shape
            h_crop, w_crop, ch_crop = cropped_img.shape
            # print(frame_rgb.shape)
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            img_crop = QImage(cropped_img_copy.data, w_crop, h_crop, ch_crop * w_crop, QImage.Format_RGB888)
            scaled_img = img.scaled(600, 480, Qt.KeepAspectRatio)
            scaled_crop_img = img_crop.scaled(155, 155, Qt.KeepAspectRatio)

            self.updateFrame.emit(scaled_img)
            self.cropFrame.emit(scaled_crop_img)
        sys.exit(-1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Add a Title and window size
        self.setWindowTitle('OpenCV Detections')
        self.setGeometry(50, 50, 800, 670)

        # Add menu bar - File
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu('File')

        # Add menu bar - About
        self.about = self.menuBar()
        self.menu_about = self.menu.addMenu('About')

        self.label1 = QLabel(self)
        # setFixedSize define o tamanho fixo do widget QLabel para 640 pixels de largura por 480 pixels de altura.
        self.label1.setFixedSize(640, 480)

        self.label_hand_crop = QLabel(self)
        self.label_hand_crop.setFixedSize(155, 155)

        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)
        self.th.cropFrame.connect(self.setCropImage)

        # Add select box layout
        trained_model_layout = QHBoxLayout()

        # Add oganized box to select element
        self.trained_group = QGroupBox('Trained Models')
        self.trained_group.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Add select options
        self.select_options = QComboBox()
        self.select_options.addItem('Random Forest')
        self.select_options.addItem('KNN')
        self.select_options.addItem('MLP')

        trained_model_layout.addWidget(QLabel('Models:'), 10)
        trained_model_layout.addWidget(self.select_options, 90)

        self.trained_group.setLayout(trained_model_layout)

        # Add configuration models
        model_config = QHBoxLayout()

        self.trained_config = QGroupBox('Model Configuration')
        self.trained_config.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        num_of_train_layout = QVBoxLayout()
        shuffle_layout = QVBoxLayout()

        self.num_of_train = QSpinBox(suffix='%')
        self.shuffle = QSpinBox(minimum=0, maximum=1)

        # Train layout
        num_of_train_layout.addWidget(QLabel('Train'), 1)
        num_of_train_layout.addWidget(self.num_of_train, 1)
        num_of_train_layout.setSpacing(2)

        # Shuffle Layout
        shuffle_layout.addWidget(QLabel('Shuffle'))
        shuffle_layout.addWidget(self.shuffle)
        shuffle_layout.setSpacing(2)

        # Main config model layout
        config_inputs = QHBoxLayout()
        config_inputs.addLayout(num_of_train_layout)
        config_inputs.addLayout(shuffle_layout)

        model_config.addWidget(QLabel('Config:'))
        model_config.addLayout(config_inputs)

        self.trained_config.setLayout(model_config)

        # Main model layout
        main_models_layout = QVBoxLayout()
        main_models_layout.addWidget(self.trained_group)
        main_models_layout.addWidget(self.trained_config)

        # Add button to start and finish
        button_layout = QVBoxLayout()
        self.bt_start = QPushButton('Start')
        self.bt_stop = QPushButton('Stop')
        # Add resposividade to buttons
        self.bt_start.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.bt_stop.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)
        # QSizePolicy.Fixed: O tamanho do widget é fixo e não será alterado ao redimensionar o layout.
        # QSizePolicy.Minimum: O tamanho mínimo do widget é definido e pode ser aumentado, mas não diminuído, ao redimensionar o layout.
        # QSizePolicy.Expanding: O widget pode expandir para ocupar espaço adicional disponível no layout, mas também pode ser reduzido caso necessário.
        # QSizePolicy.MinimumExpanding: Similar ao Expanding, mas com tamanho mínimo definido.
        # QSizePolicy.Preferred: O widget tem um tamanho preferencial, mas pode ser expandido ou reduzido de acordo com as restrições do layout.

        # Add buttons to buttons layout
        button_layout.addWidget(self.bt_start)
        button_layout.addWidget(self.bt_stop)

        # Add main control layout

        info_right = QVBoxLayout()
        info_right_box = QVBoxLayout()
        self.info_box = QGroupBox('CPU/RAM Status')
        self.info_box.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.info_box.setFixedHeight(120)

        font_cpu_percent_title = QFont()
        font_cpu_percent_title.setPointSize(18)
        font_cpu_percent = QFont()
        font_cpu_percent.setPointSize(14)

        horizontal_cpu_info = QHBoxLayout()
        horizontal_ram_info = QHBoxLayout()

        cpu_percent_title = QLabel('CPU')
        cpu_percent_title.setFont(font_cpu_percent_title)
        cpu_percent_title.setStyleSheet("color: red;")
        cpu_percent = QLabel(f'{str(psutil.cpu_percent(interval=1))}%')
        cpu_percent.setFont(font_cpu_percent)
        cpu_percent_title.setFixedSize(65, 30)
        cpu_percent.setFixedSize(65, 30)
        horizontal_cpu_info.addWidget(cpu_percent_title)
        horizontal_cpu_info.addWidget(cpu_percent)

        ram_percent_title = QLabel('RAM')
        ram_percent_title.setFont(font_cpu_percent_title)
        ram_percent_title.setStyleSheet("color: blue;")
        ram_percent = QLabel(f'{str(psutil.virtual_memory().percent)}%')
        ram_percent.setFont(font_cpu_percent)
        ram_percent_title.setFixedSize(65, 30)
        ram_percent.setFixedSize(65, 30)
        horizontal_ram_info.addWidget(ram_percent_title)
        horizontal_ram_info.addWidget(ram_percent)

        #self.label_hand_crop.setStyleSheet('background-color: red;')

        info_right_box.addLayout(horizontal_cpu_info)
        info_right_box.addLayout(horizontal_ram_info)

        self.info_box.setLayout(info_right_box)

        info_right.addWidget(self.info_box)
        info_right.addWidget(self.label_hand_crop)

        right_layout = QHBoxLayout()
        right_layout.addLayout(main_models_layout, 1)
        right_layout.addLayout(button_layout, 1)

        info_and_main_cam = QHBoxLayout()
        info_and_main_cam.addWidget(self.label1)
        info_and_main_cam.addLayout(info_right)

        """ right_imgs = QVBoxLayout()
        right_imgs.addWidget(self.label1)
        right_imgs.addWidget(self.label_hand_crop) """

        main_layout = QVBoxLayout()
        main_layout.addLayout(info_and_main_cam)
        main_layout.addLayout(right_layout)

        widget = QWidget(self)
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)

        self.bt_start.clicked.connect(self.start)
        self.bt_stop.clicked.connect(self.kill_thread)

    @Slot(QImage)
    def setImage(self, image):
        self.label1.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def setCropImage(self, image):
        self.label_hand_crop.setPixmap(QPixmap.fromImage(image))

    @Slot()
    def start(self):
        print("Starting...")
        self.th.start()

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()
        # Give time for the thread to finish
        time.sleep(1)


if __name__ == '__main__':
    app = QApplication()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
