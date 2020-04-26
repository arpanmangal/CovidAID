"""
Script to prepare combined dataset
Class 0: Normal
Class 1: Bacterial Pneumonia
Class 2: Viral Pneumonia
Class 3: COVID-19
"""
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--combine_pneumonia", action='store_true', default=False)
args = parser.parse_args()

COVID19_DATA_PATH = "./data/covid19"
COVID19_IMGS_PATH = "./covid-chestxray-dataset"
PNEUMONIDA_DATA_PATH = "./chest-xray-pneumonia"
DATA_PATH = "./data"

# Assert that the data directories are present
for d in [COVID19_DATA_PATH, COVID19_IMGS_PATH, PNEUMONIDA_DATA_PATH, DATA_PATH]:
    try:
        assert os.path.isdir(d) 
    except:
        print ("Directory %s does not exists" % d)

def create_list (split):
    assert split in ['train', 'test', 'val']

    l = []
    # Prepare list using kaggle pneumonia dataset
    for f in glob.glob(os.path.join(PNEUMONIDA_DATA_PATH, split, 'NORMAL', '*')):
        l.append((f, 0)) # Class 0

    for f in glob.glob(os.path.join(PNEUMONIDA_DATA_PATH, split, 'PNEUMONIA', '*')):
        if args.combine_pneumonia:
            l.append((f, 1)) # Class 1
        else:
            if 'bacteria' in f:
                l.append((f, 1)) # Class 1
            else:
                l.append((f, 2)) # Class 2

    # Prepare list using covid dataset
    covid_file = os.path.join(COVID19_DATA_PATH, '%s_list.txt'%split)
    with open(covid_file, 'r') as cf:
        for f in cf.readlines():
            f = os.path.join(COVID19_IMGS_PATH, f.strip())
            if args.combine_pneumonia:
                l.append((f, 2)) # Class 2
            else:
                l.append((f, 3)) # Class 3

    with open(os.path.join(DATA_PATH, '%s.txt'%split), 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)

for split in ['train', 'test', 'val']:
    create_list(split)
    
