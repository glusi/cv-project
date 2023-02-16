from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from tensorflow.image import convert_image_dtype, resize, ResizeMethod
from tqdm import tqdm
from pathlib import Path
from tensorflow import float32
from sklearn.metrics import classification_report
import tensorflow as tf
SIZE = 224

def font_to_num(font):
    if font == b'Alex Brush':
        return 0
    elif font == b'Open Sans':
        return 1
    elif font == b'Sansation':
        return 2
    elif font == b'Titillium Web':
        return 3
    else:
        return 4

def create_dirs(main_dir):
    Path(main_dir).mkdir(parents=True, exist_ok=True)
    Path(main_dir+'/Alex Brush').mkdir(parents=True, exist_ok=True)
    Path(main_dir+'/Titillium Web').mkdir(parents=True, exist_ok=True)
    Path(main_dir+'/Sansation').mkdir(parents=True, exist_ok=True)
    Path(main_dir+'/Open Sans').mkdir(parents=True, exist_ok=True)
    Path(main_dir+'/Ubuntu Mono').mkdir(parents=True, exist_ok=True)

def get_image_data(db, im, has_labels=True):
    img  = db['data'][im][:]
    txts = db['data'][im].attrs['txt']
    if has_labels:
        fonts = db['data'][im].attrs['font']  
    else: 
        fonts = None
    charBBs = db['data'][im].attrs['charBB']
    wordBBs = db['data'][im].attrs['wordBB']
    return img, fonts, txts, charBBs, wordBBs

def is_not_dot(inp):
    res= (inp != ord('.') and inp != ord(':'))
    return res

def show_results(test_y, prediction_arr):    
    labels=['Alex Brush','Open Sans','Sansation','Titillium Web','Ubuntu Mono']
    print(classification_report(test_y, prediction_arr, target_names=labels))

def prepare_img(img, bbs, index, size = SIZE):
    x1 = int(bbs[0,0,index])
    y1 = int(bbs[1,0,index])
    x2 = int(bbs[0,1,index])
    y2 = int(bbs[1,1,index])
    x3 = int(bbs[0,2,index])
    y3 = int(bbs[1,2,index])
    x4 = int(bbs[0,3,index])
    y4 = int(bbs[1,3,index])
    # calculate bounding rectangle
    top_left_x = max(0, min([x1,x2,x3,x4]))
    top_left_y = max(0, min([y1,y2,y3,y4]))
    bot_right_x = max(0, max([x1,x2,x3,x4]))
    bot_right_y = max(0, max([y1,y2,y3,y4]))

    cropped = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
    cropped = tf.image.resize(cropped, (size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped = tf.image.convert_image_dtype(cropped, tf.float32)
    return cropped

def crop_and_save(img, BBs, indx, size, curr_font, im, num, append_not_save=False, folder='main_directory/'):
    cropped = prepare_img(img, BBs, indx, size)
    if not append_not_save:
        path = folder+curr_font.decode('UTF-8')+'/'+im+'_'+str(num)+'.png' 
        tf.keras.utils.save_img(path,cropped)
    return cropped