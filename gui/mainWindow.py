import os
import sys
import time

import mediapipe as mp
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget, QLineEdit, QSpinBox)

# Parâmetros que auxiliam na ilustração/desenho dos landmarks (ponto de referência) das mãos detectadas na nossa imagem
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar o MediaPipe
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Reading the image in RGB to display it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    #Desenha os landmarks (pontos de referência) sobre as mãos da imagem, identificando a mesma e cada dedo - útil para saber onde estão os landmarks 
                    mp_drawing.draw_landmarks(
                    frame_rgb, # Image to draw
                    hand_landmarks, # model output
                    mp_hands.HAND_CONNECTIONS, # hand connection
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Creating and scaling QImage
            h, w, ch = frame_rgb.shape
            #print(color_frame.shape)
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(600, 480, Qt.KeepAspectRatio)

            self.updateFrame.emit(scaled_img)
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

        self.label = QLabel(self)
        # setFixedSize define o tamanho fixo do widget QLabel para 640 pixels de largura por 480 pixels de altura.
        self.label.setFixedSize(640, 480)

        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Add select box layout
        trained_model_layout = QHBoxLayout()

        # Add oganized box to select element
        self.trained_group = QGroupBox('Trained Models')
        self.trained_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

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
        self.trained_config.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        num_of_train_layout = QVBoxLayout()
        shuffle_layout = QVBoxLayout()

        self.num_of_train = QSpinBox(suffix='%')
        self.shuffle = QSpinBox(minimum=0, maximum=1)

        # Train layout
        num_of_train_layout.addWidget(QLabel('Train'),1)
        num_of_train_layout.addWidget(self.num_of_train,1)
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
        self.bt_start.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.bt_stop.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # QSizePolicy.Fixed: O tamanho do widget é fixo e não será alterado ao redimensionar o layout.
        # QSizePolicy.Minimum: O tamanho mínimo do widget é definido e pode ser aumentado, mas não diminuído, ao redimensionar o layout.
        # QSizePolicy.Expanding: O widget pode expandir para ocupar espaço adicional disponível no layout, mas também pode ser reduzido caso necessário.
        # QSizePolicy.MinimumExpanding: Similar ao Expanding, mas com tamanho mínimo definido.
        # QSizePolicy.Preferred: O widget tem um tamanho preferencial, mas pode ser expandido ou reduzido de acordo com as restrições do layout.
        
        # Add buttons to buttons layout
        button_layout.addWidget(self.bt_start)
        button_layout.addWidget(self.bt_stop)

        # Add main control layout

        right_layout = QHBoxLayout()
        right_layout.addLayout(main_models_layout,1)
        right_layout.addLayout(button_layout, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(right_layout)

        widget = QWidget(self)
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)

        self.bt_start.clicked.connect(self.start)
        self.bt_stop.clicked.connect(self.kill_thread)

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

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
