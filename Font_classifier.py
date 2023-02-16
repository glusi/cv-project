from sklearn.preprocessing import normalize
from utils import create_dirs, SIZE, get_image_data, crop_and_save, font_to_num
from numpy import asarray, argmax, bincount,argwhere
from h5py import File

class Font_classifier:
    def __init__(self, _model):
        self._model = _model

    def predict(self, db_file):
        db = File(db_file, 'r')   
        size=SIZE
        im_names = list(db['data'].keys())
        num = 0
        prediction_arr=[]
        images = []
        chars=[]
        for i in tqdm.tqdm(range(0, len(im_names))):
            im = im_names[i]
            img, _, txts, charBBs, wordBBs = get_image_data(db, im, no_labels=True)
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
                    chars.append(txts[j][k])
                    char_indx+=1
                test_x = asarray(test_x, dtype=np.float32)
                reses = self.model.predict(test_x, verbose=0)
                maxes = argmax(reses, axis=1)
                prediction = bincount(maxes)
                prediction = argwhere(prediction==prediction.max())
                if (len(prediction)>1):
                    sum_p = reses.sum(axis=0)
                    prediction = sum_p.argmax()
                    if(font_to_num(curr_font)!=prediction):
                        reses_n = normalize(reses, axis=1, norm='l1')
                        maxes_n = argmax(reses_n, axis=1)
                        prediction = bincount(maxes_n)
                        prediction = argmax(prediction)
                        if(font_to_num(curr_font)!=prediction):
                            sum_p = np.sum(reses, axis=0)
                            prediction = sum_p.argmax()
                for k in range(0, len(txts[j])):
                    prediction_arr.append(prediction.item())
                    images.append(im)
                font_indx += len(txts[j])
        return prediction_arr



