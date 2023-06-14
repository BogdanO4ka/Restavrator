from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
# from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from PIL import Image, ImageTk
import numpy as np
import os
import random
import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt


import tkinter
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkinter import ttk
import urllib.request

test_img = "C:/Project_school/data/20056.jpg"

window = tkinter.Tk()
window.title("Добро пожаловать в приложение нейросети")
window.geometry('500x350')

def clicked():
    dataset = filedialog.askdirectory()
    dataset_on = 1
    human = 0
    clicked = 1
    geo = 0
def human():
    dataset_on = 1
    human = 1
    clicked = 0
    geo = 0
def geo():
    dataset_on = 1
    human = 0
    clicked = 0
    geo = 1
def start():
    if dataset_on == 1:
        model.compile(optimizer='adam', loss='mse')
        model.fit(x=X, y=Y, batch_size=1, epochs=100)
        dataset_on2 = 1
def learning():
    if dataset_on2 == 1:
        model.compile(optimizer='adam', loss='mse')
        model.fit(x=X, y=Y, batch_size=1, epochs=100)
    a = 0
def save():
    a = 0
def testing():
    test_img = filedialog.askopenfilename()
    lbl5.configure(text=test_img)
def download_end():
    print('Beginning file download with urllib2...')
    url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
    urllib.request.urlretrieve(url, '/Users/scott/Downloads/cat.jpg')

lbl = tkinter.Label(window, text="Введите сюда датасет для обучения:")
lbl.grid(column=0, row=0)
btn = tkinter.Button(window, text="Ввод датасета", command=clicked)
btn.grid(column=1, row=0)
lbl2 = tkinter.Label(window, text="Или используйте встроенные:")
lbl2.grid(column=0, row=1)
btn1 = tkinter.Button(window, text="Люди", command=human)
btn1.grid(column=1, row=1)
btn2 = tkinter.Button(window, text="Местность", command=geo)
btn2.grid(column=2, row=1)

canvas = tkinter.Canvas(window, height=250, width=250)
image = Image.open("C:/Project_school/data/20056.jpg")
photo = ImageTk.PhotoImage(image)
image = canvas.create_image(0, 0, anchor='nw',image=photo)
canvas.grid(column=2, row=3)


btn3 = tkinter.Button(window, text="Запуск обработки", command=start)
btn3.grid(column=0, row=3)
btn4 = tkinter.Button(window, text="Запуск обучения", command=learning)
btn4.grid(column=0, row=5)
btn5 = tkinter.Button(window, text="Сохранить прогресс", command=save)
btn5.grid(column=0, row=6)

lbl3= tkinter.Label(window, text="-----------------------------------------------------------------------------------")
lbl3.grid(column=0, row=7)

lbl4= tkinter.Label(window, text="Вставьте фотографию для обработки:")
lbl4.grid(column=0, row=8)
btn4 = tkinter.Button(window, text="Здесь", command=testing)
btn4.grid(column=1, row=8)


lbl5= tkinter.Label(window, text="Путь изображения:")
lbl5.grid(column=0, row=10)


canvas1 = tkinter.Canvas(window, height=250, width=250)
image1 = Image.open("C:/Project_school/data/20060.jpg")
photo1 = ImageTk.PhotoImage(image1)
image1 = canvas.create_image(0, 0, anchor='nw',image=photo)
canvas1.grid(column=0, row=11)

btn5 = tkinter.Button(window, text="Скачать обработанную фотографию", command=download_end)
btn5.grid(column=1, row=8)

imgin = []

if clicked == 1:
    imgin = []
    for filename in os.listdir(dataset):
        imgin.append(img_to_array(load_img('/dataset/' + filename)))

elif human == 1:
    imgin = []
    for filename in os.listdir('/data/'):
        imgin.append(img_to_array(load_img('/data/' + filename)))

elif geo == 1:
    imgin = []
    for filename in os.listdir(''):
        imgin.append(img_to_array(load_img('/human/' + filename)))

img = Image.open(BytesIO(upl[names[0]]))


def processed_image(img):
  image = img.resize( (256, 256), Image.BILINEAR)
  image = np.array(image, dtype=float)


  size = image.shape
  lab = rgb2lab(1.0/255*image)
  X, Y = lab[:,:,0], lab[:,:,1:]

  Y /= 128
  X = X.reshape(1, size[0], size[1], 1)
  Y = Y.reshape(1, size[0], size[1], 2)
  return X, Y, size


model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

plot_model(model, show_shapes=True)

upl = files.upload()
names = list(upl.keys())
img = Image.open(BytesIO(upl[names[0]]))
X, Y, size = processed_image(img)

output = model.predict(X)

output *= 128
min_vals, max_vals = -128, 127
ab = np.clip(output[0], min_vals, max_vals)

cur = np.zeros((size[0], size[1], 3))
cur[:,:,0] = np.clip(X[0][:,:,0], 0, 100)
cur[:,:,1:] = ab
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(lab2rgb(cur))


window.mainloop()