import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



fractured = os.path.join(r'D:/Programming/Projects/Bone_Fracture/Dataset/Fractured')
train_images = os.listdir(fractured)



batch_size = 3
target_size = (100,100)
train_data = ImageDataGenerator(rescale=1/255)
train_data_gen = train_data.flow_from_directory(r'D:/Programming/Projects/Bone_Fracture/Dataset',
                                                target_size = target_size,batch_size=batch_size,
                                                classes=["Fractured","Unfractured"],class_mode='binary')
print(train_data_gen.class_indices)
#print(train_data_gen.classes)



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='tanh',input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='tanh'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
    ])
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])
model.fit(train_data_gen,steps_per_epoch=int(train_data_gen.n/batch_size),epochs=8)



model_json = model.to_json()
with open("D:/Programming/Projects/Bone_Fracture/Model/model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("D:/Programming/Projects/Bone_Fracture/Model/model1.h5")
print("Model Trained")
