import os
import keyboard
import cv2

from sklearn.neural_network import MLPClassifier

def main():
    model_mlp = MLPClassifier(max_iter=900, verbose=True, tol=0.0000100, hidden_layer_sizes=(26,26))
    print(str(model_mlp).split('(')[0])

""" def writeText(img, text, color=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    text = text.upper()
    cv2.putText(img, text, (10,25), fonte, 0.5, color, 1, cv2.LINE_AA)

def main():
    # Create the datasets path 
    #DATA_DIR = './dataSet'
    DATA_DIR = '../data/dataSetJustLeftHand'

    # Create the datasets folder
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Register the number of classes and the size to dataset
    number_of_classes = 10
    dataset_size = 100

    # Get WebCam
    cap = cv2.VideoCapture(0)
    for classes in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(classes))):
            os.makedirs(os.path.join(DATA_DIR, str(classes)))

        while True:
            res, frame = cap.read()
            if res:
                try:
                    writeText(frame, f"Get ready to register number {classes}")
                    cv2.imshow('WebCam', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break                    
                except:
                    print('ERROR')
                    break
            else:
                print('Empty WebCam')
                break

        counter = 0
        while counter < dataset_size:
            res, frame = cap.read()
            if res:
                try:
                    print(f'Dataset to number: {classes}')
                    cv2.imshow('WebCam', frame)

                    cv2.imwrite(os.path.join(DATA_DIR, str(classes), f'{counter}.jpg'), frame)

                    counter += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    print('ERROR')
                    break

            else:
                break

    cap.release()
    cv2.destroyAllWindows() """

if __name__ == '__main__':
    main()