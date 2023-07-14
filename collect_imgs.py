import os
import cv2

def writeText(img, text, color=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    text = text.upper()
    cv2.putText(img, text, (10,25), fonte, 0.5, color, 1, cv2.LINE_AA)


def main():
    # Create the data path 
    #DATA_DIR = './dataSet'
    DATA_DIR = './data/dataSetHalfBodyHand'

    # Create the data folder
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Register the number of classes and the size to dataset
    number_of_classes = 10
    dataset_size = 1
    i = 0

    # Get WebCam
    cap = cv2.VideoCapture(0)
    for classes in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(classes))):
            os.makedirs(os.path.join(DATA_DIR, str(classes)))

        for i in range(2):
            while True:
                res, frame = cap.read()
                hand = 'Right' if i == 0 else 'Left'
                if res:
                    try:
                        writeText(frame, f"Get ready to register number >>{classes}<< with {hand} hand")
                        cv2.imshow('WebCam', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break                    
                    except:
                        print('ERROR')
                        break
                else:
                    print('Empty WebCam')
                    break
            
            if len(os.listdir(os.path.join(DATA_DIR, str(classes)))) > 0:
                to_add = len(os.listdir(os.path.join(DATA_DIR, str(classes))))
            else:
                to_add = 0

            if i == 0:
                counter = 0
                compare = dataset_size//2
                print(compare)
                print(counter)
            else:
                counter = dataset_size//2
                compare = dataset_size
                print(compare)
                print(counter)
            while counter < compare:
                res, frame = cap.read()
                if res:
                    try:
                        print(f'Dataset to number: {classes}')
                        cv2.imshow('WebCam', frame)

                        cv2.imwrite(os.path.join(DATA_DIR, str(classes), f'{counter+to_add}.jpg'), frame)

                        counter += 1

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except:
                        print('ERROR')
                        break

                else:
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()