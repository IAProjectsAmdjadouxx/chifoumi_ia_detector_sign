import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import numpy as np

#model = tf.keras.models.load_model('rps.h5')


def list_file_of_dir(dir):
    import os
    list_files = os.listdir(dir)
    for file in list_files:
        print(file)

model = tf.keras.models.load_model('rps.h5')

# path = "./rps/scissors/scissors01-000.png"
# img = image.load_img(path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print(path)
# print(classes)

if __name__ == "__main__":
    while(1):
        usr_dir_choose = input("Please input the directory you want to list: ")
        list_file_of_dir("./rps-test-set/" + usr_dir_choose)

        usr_file_choose = input("Please input the file you want to predict: ")
        path = "./rps-test-set/" + usr_dir_choose + "/" + usr_file_choose
        print(path)

        #predict

        print("Do you want to continue? (y/n)")
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(path)
        print(classes)
        usr_continue = input()
        if usr_continue == "n":
            break
        else:
            continue