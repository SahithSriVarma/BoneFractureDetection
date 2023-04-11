from flask import Flask,jsonify,request
import werkzeug

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

    test_img = load_img(r"D:/Programming/Projects/Bone_Fracture/Code/image_picker2451307795387870723.jpg",target_size=(100,100,3))
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

app = Flask(__name__)
@app.route('/upload',methods = ['POST'])
def upload():
    if(request.method=="POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save(filename)
        return jsonify({"message":"Image uploaded successfully"})

@app.route('/result', methods = ['GET'])
def result():
    if request.method == 'GET':
        result = predict() 
        return jsonify({"output": result})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
