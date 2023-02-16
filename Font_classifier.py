from sklearn.preprocessing import normalize
from utils import create_dirs, SIZE, get_image_data, crop_and_save, font_to_num
from numpy import asarray, argmax, bincount,argwhere
from h5py import File
from sklearn.preprocessing import normalize
import csv
from h5py import File
from tqdm import tqdm
import numpy as np

class Font_classifier:
    def __init__(self, model):
        self.model = model

    def write_row(self, index, im, ch, prediction, writer):
        a = []
        a.append(index)
        a.append(im)
        a.append(ch)
        b = np_utils.to_categorical(prediction, 5)
        b = np.concatenate((a,b[1:],[b[0]]))
        writer.writerow(b)
        index+=1

    def predict(self, db_file):
        db = File(db_file, 'r')
        size=SIZE
        im_names = list(db['data'].keys())
        num = 0
        index = 0
        with open('test_lables.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([" ", "image", "char","Open Sans","Sansation","Titillium Web","Ubuntu Mono","Alex Brush"])
            for i in tqdm(range(0, len(im_names))):
                im = im_names[i]
                img, _, txts, charBBs, wordBBs = get_image_data(db, im, False)
                font_indx = 0 
                char_indx = 0
                for j in range(0, len(txts)):
                    test_x = [] 
                    cropped = crop_and_save(img, wordBBs, j, size, None, im, num, True)
                    test_x.append(cropped)
                    num+=1            
                    for k in range(0, len(txts[j])):
                        cropped = crop_and_save(img, charBBs, char_indx, size, None, im, num, True)
                        test_x.append(cropped)
                        num+=1
                        char_indx+=1
                    test_x = np.asarray(test_x, dtype=np.float32)
                    reses = self.model.predict(test_x, verbose=0)
                    maxes = np.argmax(reses, axis=1)
                    prediction = np.bincount(maxes)
                    prediction = np.argwhere(prediction==prediction.max())
                    if (len(prediction)>1):
                        reses_n = normalize(reses, axis=1, norm='l1')
                        maxes_n = np.argmax(reses_n, axis=1)
                        prediction = np.bincount(maxes_n)
                        prediction = np.argwhere(prediction==prediction.max())
                        if(len(prediction)>1):
                            sum_p = reses.sum(axis=0)
                            prediction = sum_p.argmax()
                    for k in range(0, len(txts[j])):
                        self.make_row(index, im, txts[j][k], prediction, writer)
                        index+=1
                    font_indx += len(txts[j])

