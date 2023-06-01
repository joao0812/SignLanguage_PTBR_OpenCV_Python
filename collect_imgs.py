import os

import cv2


def writeText(img, text, color=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    text = text.upper()
    cv2.putText(img, text, ((img.shape[0]//2)-55,35), fonte, 1.0, color, 0, 
    cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        res, frame = cap.read()
        if res:
            try:
                writeText(frame, "Testanto WebCam")
                cv2.imshow('WebCam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                print('ERROR')
                break
        else:
            print('Empty WebCam')
            break

if __name__ == '__main__':
    main()