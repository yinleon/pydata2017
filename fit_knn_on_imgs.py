import os
import glob
import time
from itertools import zip_longest
import datetime
import gc

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


img_width, img_height = 299, 299
today = datetime.datetime.now()
skip_img = np.zeros(shape=(img_width, img_height, 3))

f_out = './data/conv_feats.csv'
files = glob.glob('./data/google_images_sample/*/*/*')
knn_file = bootstrap(f_out)


def process_img(f):
    try:
        img = Image.open(f)   
            
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img.load()

            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            img = bg
            
            if not img.mode == 'RGB':
                img = img.convert('RGB')
            
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(
            img.resize(
                (img_width, img_height), 
                Image.ANTIALIAS
            )
        )

        return img_array
    
    except:
        return np.zeros(shape=(img_width, img_height, 3))


def make_resnet_conv(input_shape):
    base_model = ResNet50(input_shape=input_shape, 
                          weights='imagenet', 
                          include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    
    return base_model


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def delete_model(model, clear_session=True):
    '''removes model!
    '''
    del model
    gc.collect()
    if clear_session: K.clear_session()


def process_chunk(list_, f_out, model):
    '''
    Takes a list of filenames.
    '''
    if os.path.isfile(f_out):
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
        
    np_img = [process_img(_) for _ in list_]
    
    keep_ind = [i for i, _ in enumerate(np_img) if 
                np.array_equal(_, skip_img) == False]
    
    np_img = [_ for i, _ in enumerate(np_img) if i in keep_ind]

    if np_img:
        np_img = np.array(np_img)
    else:
        return
    
    X = preprocess_input(np_img.astype(np.float))
    
    X_conv = model.predict(X, batch_size=64)

    # compress them to 2D, new dims are d0 by d1 * d2 * d3
    train_reshape = (X_conv.shape[0], np.prod(X_conv.shape[1:]))
    
    df = pd.DataFrame(X_conv.reshape(train_reshape))
    df['filename'] = [_ for i, _ in enumerate(list_) if i in keep_ind]
    
    df.to_csv(f_out, index=False, mode=mode, header=header)

    
def bootstrap(file):
    '''
    Get filename for knn
    '''
    f = file.split('/')[-1]
    d = '/'.join(file.split('/')[:-1])
    knn_file = os.path.join(d, 'knn', f.replace('.csv', '.pkl'))
    
    return knn_file

def main():
    # transform images to convolutional features.
    base_model = make_resnet_conv((img_width, img_height, 3))
    for i, _ in enumerate(grouper(files, 1280)):
        print(i)
        if i != 0:
            process_chunk(_, f_out, base_model)
        else:
            process_chunk(_, f_out, base_model)
    delete_model(base_model)

    # read data into a dataframe, separate conv feats and filename!
    X = pd.read_csv(f_out)
    Y = X['filename']
    X_ = X[[_ for _ in X.columns if _ != 'filename']].values.astype(np.float)

    # fit the model and serialize it!
    knn = NearestNeighbors(n_neighbors=20, n_jobs=8, algorithm='ball_tree')
    knn.fit(X_)
    joblib.dump(knn, knn_file)
main()
