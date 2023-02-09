import csv

def create_output():
    with open('train_lables.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([" ", "image", "char","Open Sans","Sansation","Titillium Web","Ubuntu Mono","Alex Brush"])
        

create_output()