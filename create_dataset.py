import os

import mediapipe as mp
#O módulo mediapipe é usado para realizar análise de mídia em tempo real, fornecendo funcionalidades avançadas de detecção e rastreamento de elementos específicos, como mãos, poses e faces.

import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR ='./dataSet'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # We need converto BGR that used by OpenCV to RGB that is used by mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        # landmark = ponto de referência -> são pontos-chave que representam características específicas em uma imagem ou vídeo
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                # Draw lines in the hand
                #mp_drawing.draw_landmarks(
                    #img_rgb, # Image to draw
                    #hand_landmarks, # model output
                    #mp_hands.HAND_CONNECTIONS, # hand connection
                    #mp_drawing_styles.get_default_hand_landmarks_style(),
                    #mp_drawing_styles.get_default_hand_connections_style())

        plt.figure()
        plt.imshow(img_rgb)

plt.show()