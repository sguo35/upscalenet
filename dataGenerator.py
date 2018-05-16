import numpy as np
import keras as K
import math
from scipy import misc
import os

class DataGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=2, 
                shuffle=True):
        self.files = os.listdir('./train2017')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return abs(int(np.floor(len(self.files) / self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]
        return self.__data_generation(list_IDs_temp=list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = []
        y1 = []
        y2 = []
        y3 = []
        for file_name in list_IDs_temp:
            img = misc.imread('./train2017/' + file_name, mode='RGB')
            h, w, c = img.shape
            #print(h)
            #print(w)
            if h > 224 and w > 224:
                cropped = random_crop(img)
                #print(img)
                #print(cropped)
                x_img = misc.imresize(cropped, size=0.5)
                x_img = x_img / 255.
                # generate one hot for y
                y_img1, y_img2, y_img3 = one_hot_pixel(cropped)
                x.append(x_img)
                y1.append(y_img1)
                y2.append(y_img2)
                y3.append(y_img3)
        #print(x)
        return np.array(x), [np.array(y1), np.array(y2), np.array(y3)]

def random_crop(image):
    height, width, c = image.shape
    dy, dx = 64, 64
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[y:(y+dy), x:(x+dx), :]

def one_hot_pixel(img):
    new_img_r = np.zeros(shape=(64, 64, 64), dtype=np.float32)
    new_img_g = np.zeros(shape=(64, 64, 64), dtype=np.float32)
    new_img_b = np.zeros(shape=(64, 64, 64), dtype=np.float32)

    for y in range(64):
        for x in range(64):
            new_img_r[y][x][int(img[y][x][0] / 4)] = 1.
    for y in range(64):
        for x in range(64):
            new_img_g[y][x][int(img[y][x][1] / 4)] = 1.
    for y in range(64):
        for x in range(64):
            new_img_b[y][x][int(img[y][x][2] / 4)] = 1.
    return new_img_r, new_img_g, new_img_b