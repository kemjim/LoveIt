import numpy as np
from flask import Flask, request, Response, render_template, jsonify, redirect
import pandas as pd
from PIL import Image
import cv2
import matplotlib.image as img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow import keras
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

#route 1 : this is creating the homepage
@app.route('/') #this is creating a creator and right below the decorater we need to pass a function
def homepage():
    # returning the homepage html
    return render_template('homepage.html', methods=["GET", "POST"])


#route 2 : this is creating route of the homepage for men shirts category
@app.route('/form_men_shirts_hp', methods=["GET", "POST"])
def form_men_shirts_hp():
    # returning the homepage html template for men shirts category
    return render_template("form_men_shirts_hp.html")


#route 3 : this is creating the route for men shirts category - step 1 process
@app.route('/form_men_shirts_liked', methods=["GET", "POST"])
def form_men_shirts_liked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]

            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/men_shirts_liked/"

            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template for men shirts category - step 1
    return render_template('form_men_shirts_liked.html')


#route 4 : this is creating the route for men shirts category - step 2 process
@app.route('/form_men_shirts_disliked', methods=["GET", "POST"])
def form_men_shirts_disliked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]

            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/men_shirts_disliked/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template for men shirts category - step 2
    return render_template('form_men_shirts_disliked.html')


#route 5 : this is creating the route for men shirts category - step 3 process
@app.route('/form_men_shirts_check', methods=["GET", "POST"])
def form_men_shirts_check():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/men_shirts_check/"
            # saving the image into the given path.
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            # creating a list of photos from liked folder - using .os library
            men_shirts_liked_photos = os.listdir("static/new_files/men_shirts_liked/")
            try:
                # if there is .DS_Store element in the list remove it.(there is almost everytime)
                men_shirts_liked_photos.remove('.DS_Store')
            except:
                pass

            # creating 2 lists. One is for photo data and the other one is for target where we will keep 1
            men_shirts_liked_photo_data = []
            men_shirt_liked_target_data = []
            # for each item in liked photos
            for i in men_shirts_liked_photos:
                # add "1" in the target list
                men_shirt_liked_target_data.append(1)
                # also pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/men_shirts_liked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                men_shirts_liked_photo_data.append(string)

            # creating a list of photos from disliked folder - using .os library
            men_shirts_disliked_photos = os.listdir("static/new_files/men_shirts_disliked/")
            try:
                # if there is .DS_Store element in the list remove it.
                men_shirts_disliked_photos.remove('.DS_Store')
            except:
                pass
            # creating 2 lists. One is for photo data and the other one is for target where we will keep 0
            men_shirts_disliked_photo_data = []
            men_shirt_disliked_target_data = []

            # for each item in liked photos
            for i in men_shirts_disliked_photos:
                # add "1" in the target list
                men_shirt_disliked_target_data.append(0)
                # pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/men_shirts_disliked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                men_shirts_disliked_photo_data.append(string)

            # our model takes array. So we merge liked and disliked lists and convert them into an array.
            # we name the photo data array as X_train, and target array as y_train array
            X_train = np.array(men_shirts_liked_photo_data + men_shirts_disliked_photo_data)
            y_train = np.array(men_shirt_liked_target_data + men_shirt_disliked_target_data)
            # The model performs better with multiclass categorical target. So, we convert y_train into categorical array using Kera's .utils function
            y_train = utils.to_categorical(y_train)

            # the model was saved as "men_shirts_machine". We are calling it back here.
            men_shirts_model = keras.models.load_model("machine_3")
            # fit the model on our training data
            men_shirts_model.fit(X_train, y_train, batch_size=8, epochs=20, verbose=0)
            # making predictions based on the photo from "men_shirts_check" file. We use .predict model and take the result with max probibility using .np.argmax. (this was recommended by Keras)
            the_machines_opinion = np.argmax(men_shirts_model.predict(cv2.resize(img.imread("static/new_files/men_shirts_check/"+image.filename)/255, (1196,780), interpolation=cv2.INTER_AREA).reshape(-1, 1196,780, 3)), axis=-1)
            # If the opinion is 1, the shirt will be liked. If the opinion is not 1, it will not be liked.
            if the_machines_opinion == 1:
                the_opinion = "Congratulations! They will LoveIt!"
            else:
                the_opinion = " This is ok, maybe try something different"
            #show the result at results.html
            return render_template('results.html', prediction=the_opinion)

    # returning the html template for men shirts category - step 3
    return render_template('form_men_shirts_check.html')


#route 6 : this is creating the route for women shirts category homepage
@app.route('/form_women_shirts_hp', methods=["GET", "POST"])
def form_women_shirts_hp():
    # returning the html template
    return render_template("form_women_shirts_hp.html")


#route 7: this is creating the route for women shirts category -  step 1
@app.route('/form_women_shirts_liked', methods=["GET", "POST"])
def form_women_shirts_liked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_shirts_liked/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    # returning the html template
    return render_template('form_women_shirts_liked.html')


#route 8: this is creating the route for women shirts category -  step 2
@app.route('/form_women_shirts_disliked', methods=["GET", "POST"])
def form_women_shirts_disliked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_shirts_disliked/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template
    return render_template('form_women_shirts_disliked.html')


#route 9: this is creating the route for women shirts category -  step 3
@app.route('/form_women_shirts_check', methods=["GET", "POST"])
def form_women_shirts_check():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]

            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_shirts_check/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            # creating a list of photos from liked folder - using .os library
            women_shirts_liked_photos = os.listdir("static/new_files/women_shirts_liked/")
            try:
                # if there is .DS_Store element in the list remove it.(there is almost everytime)
                women_shirts_liked_photos.remove('.DS_Store')
            except:
                pass

            # creating 2 lists. One is for photo data and the other one is for target where we will keep 1
            women_shirts_liked_photo_data = []
            women_shirt_liked_target_data = []
            # for each item in liked photos
            for i in women_shirts_liked_photos:
                # add "1" in the target list
                women_shirt_liked_target_data.append(1)
                # also pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/women_shirts_liked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                women_shirts_liked_photo_data.append(string)

            # creating a list of photos from disliked folder - using .os library
            women_shirts_disliked_photos = os.listdir("static/new_files/women_shirts_disliked/")
            try:
                # if there is .DS_Store element in the list remove it.
                women_shirts_disliked_photos.remove('.DS_Store')
            except:
                pass
            # creating 2 lists. One is for photo data and the other one is for target where we will keep 0
            women_shirts_disliked_photo_data = []
            women_shirt_disliked_target_data = []

            # for each item in liked photos
            for i in women_shirts_disliked_photos:
                # add "1" in the target list
                women_shirt_disliked_target_data.append(0)
                # pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/women_shirts_disliked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                women_shirts_disliked_photo_data.append(string)

            # our model takes array. So we merge liked and disliked lists and convert them into an array.
            # we name the photo data array as X_train, and target array as y_train array
            X_train = np.array(women_shirts_liked_photo_data + women_shirts_disliked_photo_data)
            y_train = np.array(women_shirt_liked_target_data + women_shirt_disliked_target_data)
            # The model performs better with multiclass categorical target. So, we convert y_train into categorical array using Kera's .utils function
            y_train = utils.to_categorical(y_train)

            # We are calling the machine back here. (It was saved)
            women_shirts_model = keras.models.load_model("machine_3")
            # fit the model on our training data
            women_shirts_model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=0)
            # making predictions based on the photo from "women_shirts_check" file. We use .predict model and take the result with max probibility using .np.argmax. (this was recommended by Keras)
            the_machines_opinion = np.argmax(women_shirts_model.predict(cv2.resize(img.imread("static/new_files/women_shirts_check/"+image.filename)/255, (1196,780), interpolation=cv2.INTER_AREA).reshape(-1, 1196,780, 3)), axis=-1)
            # If the opinion is 1, the shirt will be liked. If the opinion is not 1, it will not be liked.
            if the_machines_opinion == 1:
                the_opinion = "Congratulations! They will LoveIt!"
            else:
                the_opinion = " This is ok, maybe try something different"
            #show the result at results.html
            return render_template('results.html', prediction=the_opinion)

    # returning the html template
    return render_template('form_women_shirts_check.html')


#route 10: this is creating the route for women dress category homepage
@app.route('/form_women_dress_hp', methods=["GET", "POST"])
def form_women_dress_hp():
    # returning the html template
    return render_template("form_women_dress_hp.html")


#route 11: this is creating the route for women dress category -  step 1
@app.route('/form_women_dress_liked', methods=["GET", "POST"])
def form_women_dress_liked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_dress_liked/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template
    return render_template('form_women_dress_liked.html')


#route 12: this is creating the route for women dress category -  step 2
@app.route('/form_women_dress_disliked', methods=["GET", "POST"])
def form_women_dress_disliked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_dress_disliked/"
            # saving the image
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template
    return render_template('form_women_dress_disliked.html')


#route 13: this is creating the route for women dress category -  step 3
@app.route('/form_women_dress_check', methods=["GET", "POST"])
def form_women_dress_check():

    if request.method == "POST":

        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/women_dress_check/"
            # saving the data
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            # creating a list of photos from liked folder - using .os library
            women_dress_liked_photos = os.listdir("static/new_files/women_dress_liked/")
            try:
                # if there is .DS_Store element in the list remove it.(there is almost everytime)
                women_dress_liked_photos.remove('.DS_Store')
            except:
                pass

            # creating 2 lists. One is for photo data and the other one is for target where we will keep 1
            women_dress_liked_photo_data = []
            women_dress_liked_target_data = []
            # for each item in liked photos
            for i in women_dress_liked_photos:
                # add "1" in the target list
                women_dress_liked_target_data.append(1)
                # also pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/women_dress_liked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                women_dress_liked_photo_data.append(string)

            # creating a list of photos from disliked folder - using .os library
            women_dress_disliked_photos = os.listdir("static/new_files/women_dress_disliked/")
            try:
                # if there is .DS_Store element in the list remove it.
                women_dress_disliked_photos.remove('.DS_Store')
            except:
                pass
            # creating 2 lists. One is for photo data and the other one is for target where we will keep 0
            women_dress_disliked_photo_data = []
            women_dress_disliked_target_data = []

            # for each item in liked photos
            for i in women_dress_disliked_photos:
                # add "1" in the target list
                women_dress_disliked_target_data.append(0)
                # pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/women_dress_disliked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                women_dress_disliked_photo_data.append(string)

            # our model takes array. So we merge liked and disliked lists and convert them into an array.
            # we name the photo data array as X_train, and target array as y_train array
            X_train = np.array(women_dress_liked_photo_data + women_dress_disliked_photo_data)
            y_train = np.array(women_dress_liked_target_data + women_dress_disliked_target_data)
            # The model performs better with multiclass categorical target. So, we convert y_train into categorical array using Kera's .utils function
            y_train = utils.to_categorical(y_train)

            # We are calling the model back here.
            women_dress_model = keras.models.load_model("machine_3")
            # fit the model on our training data
            women_dress_model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=0)
            # making predictions based on the photo from "women_dress_check" file. We use .predict model and take the result with max probibility using .np.argmax. (this was recommended by Keras)
            the_machines_opinion = np.argmax(women_dress_model.predict(cv2.resize(img.imread("static/new_files/women_dress_check/"+image.filename)/255, (1196,780), interpolation=cv2.INTER_AREA).reshape(-1, 1196,780, 3)), axis=-1)
                # If the opinion is 1, the shirt will be liked. If the opinion is not 1, it will not be liked.
            if the_machines_opinion == 1:
                the_opinion = "Congratulations! They will LoveIt!"
            else:
                the_opinion = " This is ok, maybe try something different"

                #show the result at results.html
            return render_template('results.html', prediction=the_opinion)
    # returning the html template
    return render_template('form_women_dress_check.html')


#route 14: this is creating the route for shoes category homepage
@app.route('/form_shoes_hp', methods=["GET", "POST"])
def form_shoes_hp():
        # returning the html template
    return render_template("form_shoes_hp.html")


#route 15: this is creating the route for shoes category - step 1
@app.route('/form_shoes_liked', methods=["GET", "POST"])
def form_shoes_liked():

    if request.method == "POST":
        # pulling the images
        if request.files:
            image = request.files["image"]
            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/shoes_liked/"
            # saving the image data
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    # returning the html template
    return render_template('form_shoes_liked.html')


#route 16: this is creating the route for shoes category - step 2
@app.route('/form_shoes_disliked', methods=["GET", "POST"])
def form_shoes_disliked():

    if request.method == "POST":

        # pulling the images
        if request.files:
            image = request.files["image"]

            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/shoes_disliked/"

            # saving the image data
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

    # returning the html template
    return render_template('form_shoes_disliked.html')


#route 17: this is creating the route for shoes category - step 3
@app.route('/form_shoes_check', methods=["GET", "POST"])
def form_shoes_check():

    if request.method == "POST":

        if request.files:

            # pulling the images
            image = request.files["image"]

            # creating the path
            app.config["IMAGE_UPLOADS"] = "/Users/kemalcanalaeddinoglu/Desktop/LoveIt/static/new_files/shoes_check/"

            # saving the image data
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            # creating a list of photos from liked folder - using .os library
            shoes_liked_photos = os.listdir("static/new_files/shoes_liked/")
            try:
                # if there is .DS_Store element in the list remove it.(there is almost everytime)
                shoes_liked_photos.remove('.DS_Store')
            except:
                pass

            # creating 2 lists. One is for photo data and the other one is for target where we will keep 1
            shoes_liked_photo_data = []
            shoes_liked_target_data = []
            # for each item in liked photos
            for i in shoes_liked_photos:
                # add "1" in the target list
                shoes_liked_target_data.append(1)
                # also pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/shoes_liked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                shoes_liked_photo_data.append(string)

            # creating a list of photos from disliked folder - using .os library
            shoes_disliked_photos = os.listdir("static/new_files/shoes_disliked/")
            try:
                # if there is .DS_Store element in the list remove it.
                shoes_disliked_photos.remove('.DS_Store')
            except:
                pass
            # creating 2 lists. One is for photo data and the other one is for target where we will keep 0
            shoes_disliked_photo_data = []
            shoes_disliked_target_data = []

            # for each item in liked photos
            for i in shoes_disliked_photos:
                # add "1" in the target list
                shoes_disliked_target_data.append(0)
                # pull the photo data for the item and convert it to float and divide the values by 255
                string = img.imread("static/new_files/shoes_disliked/"+i).astype('float32')/255
                # resize that photo value to 780 x1196
                string = cv2.resize(string, (780, 1196), interpolation = cv2.INTER_AREA)
                # finally append it to the data list where we keep the photo data
                shoes_disliked_photo_data.append(string)

            # our model takes array. So we merge liked and disliked lists and convert them into an array.
            # we name the photo data array as X_train, and target array as y_train array
            X_train = np.array(shoes_liked_photo_data + shoes_disliked_photo_data)
            y_train = np.array(shoes_liked_target_data + shoes_disliked_target_data)
            # The model performs better with multiclass categorical target. So, we convert y_train into categorical array using Kera's .utils function
            y_train = utils.to_categorical(y_train)

            # We are calling the machine back here.
            shoes_model = keras.models.load_model("machine_3")
            # fit the model on our training data
            shoes_model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=0)
            # making predictions based on the photo from "shoes_check" file. We use .predict model and take the result with max probibility using .np.argmax. (this was recommended by Keras)
            the_machines_opinion = np.argmax(shoes_model.predict(cv2.resize(img.imread("static/new_files/shoes_check/"+image.filename)/255, (1196,780), interpolation=cv2.INTER_AREA).reshape(-1, 1196,780, 3)), axis=-1)
            # If the opinion is 1, the shirt will be liked. If the opinion is not 1, it will not be liked.
            if the_machines_opinion == 1:
                the_opinion = "Congratulations! They will LoveIt!"
            else:
                the_opinion = " This is ok, maybe try something different"

            #show the result at results.html
            return render_template('results.html', prediction=the_opinion)

    # returning the html template
    return render_template('form_shoes_check.html')




if __name__ == "__main__": #if you run "python app_starter.py" from the terminal then the line above will resolve to True
    app.run(debug=True)
