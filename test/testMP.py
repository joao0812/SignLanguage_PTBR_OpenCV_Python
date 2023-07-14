import os
import keyboard
import cv2

from sklearn.neural_network import MLPClassifier

def main():
    to_add = 100
    DATA_DIR = './data/dataSetHalfBodyHand/'
    print(os.listdir(DATA_DIR))
    img_dir_0 = os.path.join(DATA_DIR, '0')
    list_img_0_dir = os.listdir(img_dir_0)

    if len(list_img_0_dir) > 0:
        for i in range(to_add):
            print(f'{len(list_img_0_dir)+i}.jpg')

    """ for dir in os.listdir(os.path.join(DATA_DIR, '0')):
        print(dir) """

if __name__ == '__main__':
    main()