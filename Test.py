import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from matplotlib import pyplot as plt


def predict():
    json_model = open('D:/Programming/Projects/Bone_Fracture/Model/model.json','r')
    loaded_json = json_model.read()
    json_model.close()

    loaded_model = tf.keras.models.model_from_json(loaded_json)
    loaded_model.load_weights("D:/Programming/Projects/Bone_Fracture/Model/model1.h5")
    print("Model Loaded")

    test_img = load_img(r"D:/Programming/Projects/Bone_Fracture/TestData/test1.jpg",target_size=(100,100,3))
    print("Image Loaded")
    plt.imshow(test_img)

    test_img = img_to_array(test_img)
    test_img = np.expand_dims(test_img,axis=0)
    test_img = np.vstack([test_img])

    result = loaded_model.predict(test_img)
    if result <= 0.5:
        return("Fractured")
    else:
        return("Unfractured")
