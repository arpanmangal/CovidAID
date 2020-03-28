"""
Script to prepare data
"""
import glob
import os

covid_path = "../covid19_xray"
pneumonia_path = "../chest-xray-pneumonia"

def create_list (split):
    assert split in ['train', 'test', 'val']

    l = []
    # Prepare list using kaggle pneumonia dataset
    for f in glob.glob(os.path.join(pneumonia_path, split, 'NORMAL', '*')):
        l.append((f, 0)) # Class 0

    for f in glob.glob(os.path.join(pneumonia_path, split, 'PNEUMONIA', '*')):
        if 'bacteria' in f:
            l.append((f, 1)) # Class 1
        else:
            l.append((f, 2)) # Class 2

    # Prepare list using covid dataset
    covid_file = os.path.join(covid_path, '%s.txt'%split)
    with open(covid_file, 'r') as cf:
        for f in cf.readlines():
            f = os.path.join(covid_path, 'xray_covid_images', f.strip())
            l.append((f, 3)) # Class 3

    with open('data/%s.txt'%split, 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)

for split in ['train', 'test', 'val']:
    create_list(split)
    
