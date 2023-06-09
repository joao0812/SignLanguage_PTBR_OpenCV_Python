import os, pickle
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

def writeText(img, text, color=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    text = text.upper()
    cv2.putText(img, text, (10,25), fonte, 1.0, color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (11,26), fonte, 1.0, color, 1, cv2.LINE_AA)

# Parâmetros que auxiliam na ilustração/desenho dos landmarks (ponto de referência) das mãos detectadas na nossa imagem
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}

while True:

    data_aux = []

    #model_dict = pickle.load(open('./model.p', 'rb'))
    model_dict = pickle.load(open('./modelHalfBodyHand.p', 'rb'))
    model = model_dict['model']

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

                    data_aux.append(x)
                    data_aux.append(y)

        #print(len(data_aux))
        prediction = model.predict(np.asarray([data_aux]))
        
        print(prediction[0])
        predicted_character = labels_dict[prediction[0]]

        #print(predicted_character)
        writeText(frame, f'Number: {prediction[0]}')
    else:
        writeText(frame, 'Number: NaN',(0,0,255)) 
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()