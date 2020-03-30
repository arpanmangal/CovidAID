"""
Script for preparing COVID=19 data, and performing train-val-test split
"""

import csv
import os
import numpy as np

# Dataset path
COVID_DATA_PATH='./covid-chestxray-dataset'
METADATA_PATH = os.path.join(COVID_DATA_PATH, 'metadata.csv')
IMAGES_PATH = os.path.join(COVID_DATA_PATH, 'images')
TRAIN_FILE = './data/covid/train_list.txt'
VAL_FILE = './data/covid/val_list.txt'
TEST_FILE = './data/covid/test_list.txt'

# Load patient stats
covids = dict()
with open(METADATA_PATH, newline='') as csvfile:
    covidreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in covidreader:
        patient_id = row[0]
        finding = row[4]
        view = row[11]
        modality = row[12]
        if finding == 'COVID-19' and modality == 'X-ray' and view in ['PA', 'AP', 'AP Supine']:
            if patient_id in covids:
                covids[patient_id] += 1
            else:
                covids[patient_id] = 1

covids = {k: v for k, v in sorted(covids.items(), key=lambda item: item[1])}
total_data = sum([v for k,v in covids.items()])
print ("Patient-#X-Rays statistics:")
print (covids)
print ("Total Images:", total_data, '\n')

# Assign train-val-test split
test_patients = set({4, 15, 86, 59, 6, 82, 80, 78, 76, 65, 36, 32, 50, 18})
val_patients = set({73, 51, 48, 11, 43, 24, 112})

print ('#Train data points:', sum([v for k, v in covids.items() if int(k) not in test_patients.union(val_patients)]))
print ('#Test data points:', sum([v for k, v in covids.items() if int(k) in test_patients]))
print ('#Val data points:', sum([v for k, v in covids.items() if int(k) in val_patients]))


# Construct the split lists
train_list = []
test_list = []
val_list = []

with open(METADATA_PATH, newline='') as csvfile:
    covidreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in covidreader:
        patient_id = row[0]
        finding = row[4]
        view = row[11]
        modality = row[12]
        filename = row[15]
        if finding == 'COVID-19' and modality == 'X-ray' and view in ['PA', 'AP', 'AP Supine']:
            if int(patient_id) in test_patients:
                test_list.append(filename)
            elif int(patient_id) in val_patients:
                val_list.append(filename)
            else:
                train_list.append(filename)

print (len(train_list), len(test_list), len(val_list))


# Write image list in file
def make_img_list(data_file, img_file_list):
    with open(data_file, 'w') as f:
        for imgfile in img_file_list:
            try: 
                assert os.path.isfile(os.path.join(IMAGES_PATH, imgfile))
                f.write("%s\n" % imgfile)
            except: 
                print ("Image %s NOT FOUND" % imgfile)

make_img_list(TRAIN_FILE, train_list)
make_img_list(VAL_FILE, val_list)
make_img_list(TEST_FILE, test_list)
