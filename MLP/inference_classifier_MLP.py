import os, pickle
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def writeText(img, text, color=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    text = text.upper()
    cv2.putText(img, text, (10,25), fonte, 1.0, color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (11,26), fonte, 1.0, color, 1, cv2.LINE_AA)

def cropImage(img, cord1, cord2, class_pred, class_porcent):
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
    
    #print(min_x)
    #print(min_y)
    #print(max_x)
    #print(max_y)
    
    cropped_img = img_copy[min_y:max_y, min_x:max_x]
    #cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (255,0,0), 5)
    #cv2.putText(img_copy, f'{class_pred[0]} -- {(class_porcent*100):.2f}%', (min_x,min_y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,0), 1, cv2.LINE_AA)
    return cropped_img


def defineHand(img, cord1, cord2, result):
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

    handedness_list = result.handedness
    print(handedness_list[0])


    cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (255,0,0), 5)
    cv2.putText(img_copy, f'aa', (min_x,min_y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,0), 1, cv2.LINE_AA)
    return img_copy

# Parâmetros que auxiliam na ilustração/desenho dos landmarks (ponto de referência) das mãos detectadas na nossa imagem
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

labels_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
min_porcent = 0.998

while True:

    data_aux = []
    aux_cord_pixel = []
    cord_pixel = []

    #model_dict = pickle.load(open('./model.p', 'rb'))
    model_dict = pickle.load(open('./models/MPLClassifier.p', 'rb'))
    model = model_dict['model_mlp']

    res, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            #Desenha os landmarks (pontos de referência) sobre as mãos da imagem, identificando a mesma e cada dedo - útil para saber onde estão os landmarks 
            mp_drawing.draw_landmarks(
            frame, # Image to draw
            hand_landmarks, # model output
            mp_hands.HAND_CONNECTIONS, # hand connection
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                if len(hand_landmarks.landmark)<= 42:
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    aux_cord_pixel.append(x*w)
                    aux_cord_pixel.append(y*h)

                    cord_pixel.append(aux_cord_pixel)

                    data_aux.append(x)
                    data_aux.append(y)
                    aux_cord_pixel = []

        #print(cord_pixel)

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

        cropped_img = cropImage(frame, min_points, max_points, prediction, prediction_porcent)
        cv2.imshow('Cropped image', cropped_img)
        
        print(prediction[0])
        print(f'Predições porcentagem: {prediction_porcent}')
        predicted_character = labels_dict[prediction[0]]

        #print(predicted_character)
        if prediction_porcent.max() >= min_porcent:
            writeText(frame, f'Number: {prediction[0]} - {(prediction_porcent*100):.2f}%')
        else:
            writeText(frame, 'Low porcent')
    else:
        writeText(frame, 'Number: NaN',(0,0,255)) 

    
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()