import csv
from Font_classifier import Font_classifier
from download_h5py_db import download_file_from_google_drive
from tqdm import tqdm
from tensorflow.keras.models import load_model
from pathlib import Path
from utils import show_results
from h5py import File

def load_classifier_model(path):
    model = load_model(path)
    return Font_classifier(model)


def create_output():
    with open('train_lables.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([" ", "image", "char","Open Sans","Sansation","Titillium Web","Ubuntu Mono","Alex Brush"])
        for
        
FILE_NAME = 'test_data.h5'
if not Path('test_data.h5').exists():
    download_file_from_google_drive('1YwLcXqLArFSOtoepQw7nC1t4jC8CFxpI', FILE_NAME)
db = File(FILE_NAME, 'r')
path = 'res//top_model.h5'
model = load_model(path)
model = load_classifier_model(model)
pred_y, test_y = model.predict(FILE_NAME)
show_results(pred_y, test_y)