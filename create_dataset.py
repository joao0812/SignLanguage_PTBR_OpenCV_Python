#O módulo pickle é uma biblioteca em Python que permite serializar e desserializar objetos Python. A serialização é o processo de converter um objeto Python em uma sequência de bytes, que pode ser gravada em um arquivo ou transmitida por uma rede. A desserialização é o processo inverso, em que a sequência de bytes é convertida novamente em um objeto Python. O módulo pickle é amplamente utilizado para salvar e carregar objetos Python de forma persistente, ou seja, para armazená-los em disco para uso futuro. Ele é especialmente útil quando você precisa salvar estruturas de dados complexas, como listas, dicionários, classes e objetos personalizados.
import os, pickle

#O módulo mediapipe é usado para realizar análise de mídia em tempo real, fornecendo funcionalidades avançadas de detecção e rastreamento de elementos específicos, como mãos, poses e faces.
import mediapipe as mp

import cv2
import matplotlib.pyplot as plt

# Parâmetros que auxiliam na ilustração/desenho dos landmarks (ponto de referência) das mãos detectadas na nossa imagem
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#DATA_DIR ='./dataSet'
DATA_DIR = './dataSetHalfBodyHand'

# Tem todos os dados necessários do dataset
data = [] # Contem a classificação das imagens
labels = [] # Contem a categorização das imagens

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # Array temporário para auxiliar o append corredo em data
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # We need converto BGR that used by OpenCV to RGB that is used by mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        # landmark = ponto de referência -> são pontos-chave que representam características específicas em uma imagem ou vídeo
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                # Desenha os landmarks (pontos de referência) sobre as mãos da imagem, identificando a mesma e cada dedo - útil para saber onde estão os landmarks 
                #mp_drawing.draw_landmarks(
                    #img_rgb, # Image to draw
                    #hand_landmarks, # model output
                    #mp_hands.HAND_CONNECTIONS, # hand connection
                    #mp_drawing_styles.get_default_hand_landmarks_style(),
                    #mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i]) # -> Retorna um conjunto de coordenadas x, y e z de cada landmark - para o projeto usamos apenas x e y (horizontal e vertical) - E com esses dados vamos gerar um longo array para treinar a classificação, ou seja, vamos coletar esses dados e dizer a qual classe ele representa
                    #As coordenadas X e Y são normalizadas e representam as posições do ponto de referência em relação à largura e altura da imagem ou quadro de vídeo. Elas variam entre 0 e 1, em que 0 representa a posição mais à esquerda ou no topo, e 1 representa a posição mais à direita ou na parte inferior. A coordenada Z representa a distância do ponto de referência em relação à câmera. No entanto, essa coordenada só se torna útil se você estiver trabalhando com dados 3D.
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            data.append(data_aux) # Tem um par de coordenadas x, y que representa os dados de uma mão detectada 
            labels.append(dir_) # Adicionamos as classes (as letras e números), esse dir_ tem o nome das pastas em DATA_DIR
            # Com isso, se temos um conjunto informando q tão coordenada representa uma determinada classe

#print(len(data))
# Criando um arquivo que vai armazenar um dict que contem os arrays data e labels para não precisar fazer todo o processo de de registro dos dados em cada array, a fim de otimizar o processo
f = open('dataHalfBodyHand.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f) # Dataset
f.close()