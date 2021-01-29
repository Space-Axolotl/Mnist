import tensorflow as tf
from cv2 import cv2
import numpy as np

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_array.reshape(-1, 784)



model = tf.keras.models.load_model("/home/user/folder/mnisTrained.model")

prediction = model.predict((prepare('/home/user/folder/filename.png')))
print("")
print("")
print("")
print("")
print("")
print("I think this photo represents number: ")
print(CATEGORIES[np.argmax(prediction[0])])
