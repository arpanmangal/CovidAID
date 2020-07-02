
"""
Script for preparing COVID=19 data, and performing train-val-test split
"""

# import csv
import os
import numpy as np
import pandas as pd
from collections import Counter 
import argparse
np.random.seed(0)


# Dataset path
COVID_DATA_PATH='/home/cse/dual/cs5150296/scratch/COVID_Data/covid-chestxray-dataset'
BSTI_DATA_PATH='/home/cse/dual/cs5150296/scratch/COVID_Data/BSTI'
KGP_DATA_PATH='/home/cse/dual/cs5150296/scratch/COVID_Data/IITKGP-Action/images/'
METADATA_CSV = os.path.join(COVID_DATA_PATH, 'metadata.csv')
TRAIN_FILE = './data/covid19/train_list.txt'
VAL_FILE = './data/covid19/val_list.txt'
TEST_FILE = './data/covid19/test_list.txt'
BSTI_TRAIN_FILE = './data/covid19/bsti_train_list.txt'
BSTI_VAL_FILE = './data/covid19/bsti_val_list.txt'
BSTI_TEST_FILE = './data/covid19/bsti_test_list.txt'
KGP_TRAIN_FILE = './data/covid19/kgp_train_list.txt'
KGP_VAL_FILE = './data/covid19/kgp_val_list.txt'
KGP_TEST_FILE = './data/covid19/kgp_test_list.txt'
REMOVED_LIST = './data/covid19/removed_files.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--bsti", action='store_true', default=False)
parser.add_argument("--kgp_action", action='store_true', default=False)
args = parser.parse_args()

# Load patient stats
covids = dict()
df = pd.read_csv(METADATA_CSV)
df = df[(df['finding'] == 'COVID-19') & (df['modality'] == 'X-ray') & (
                (df['view'] == 'PA') | (df['view'] == 'AP') | (df['view'] == 'AP Supine')
            )]

patient_ids = Counter(df['patientid'].tolist())
covids = {k: v for k, v in sorted(patient_ids.items(), key=lambda item: item[1])}
total_data = sum([v for k,v in covids.items()])
print ("Patient-#X-Rays statistics:")
print (covids)
print ("Total Images:", total_data, '\n')

# Assign train-val-test split
test_patients = set({4, 15, 86, 59, 6, 82, 80, 78, 76, 65, 36, 32, 50, 18, 115, 152, 138, 70, 116, 121, 133, 136, 139, 144, 154, 161, 163, 165})
val_patients = set({73, 51, 48, 11, 43, 24, 112, 181})

removed_files = set()

with open(REMOVED_LIST, 'r') as removed_list:
        for filename in removed_list:
            filename = filename.rstrip()
            removed_files.add(filename)

#Initial values for covid-chestxray-dataset prior to removal
print ('#Train patients:', len(set(covids.keys()).difference(test_patients.union(val_patients))))
print ('#Test patients:', len(test_patients))
print ('#Val patients:', len(val_patients))
print ()
print ('#Train data points:', sum([v for k, v in covids.items() if int(k) not in test_patients.union(val_patients)]))
print ('#Test data points:', sum([v for k, v in covids.items() if int(k) in test_patients]))
print ('#Val data points:', sum([v for k, v in covids.items() if int(k) in val_patients]))

# Construct the split lists
train_list = []
test_list = []
val_list = []

for i, row in df.iterrows():
    patient_id = row['patientid']
    if row['filename'] in removed_files:
        continue
    filename = os.path.join(row['folder'], row['filename'])

    if int(patient_id) in test_patients:
        test_list.append(filename)
    elif int(patient_id) in val_patients:
        val_list.append(filename)
    else:
        train_list.append(filename)

print ("covid-chestxray-dataset train-test-val split: ",len(train_list), len(test_list), len(val_list))

# Write image list in file
def make_img_list(data_file, img_file_list, data_path):
    with open(data_file, 'w') as f:
        for imgfile in img_file_list:
            try: 
                assert os.path.isfile(os.path.join(data_path, imgfile))
                f.write("%s\n" % imgfile)
            except: 
                print ("Image %s NOT FOUND" % imgfile)

make_img_list(TRAIN_FILE, train_list, COVID_DATA_PATH)
make_img_list(VAL_FILE, val_list, COVID_DATA_PATH)
make_img_list(TEST_FILE, test_list, COVID_DATA_PATH)

#Include BSTI Dataset
if args.bsti :
    # Construct the split lists
    bsti_train_list = []
    bsti_test_list = []
    bsti_val_list = []
  
    bsti_images = [f for f in os.listdir(BSTI_DATA_PATH) if os.path.isfile(os.path.join(BSTI_DATA_PATH, f))]
    for imgfile in bsti_images:
        rand_val = np.random.rand(1)
        if rand_val < 0.1:
            bsti_val_list.append(imgfile)
        elif rand_val < 0.3:
            bsti_test_list.append(imgfile)
        else:
            bsti_train_list.append(imgfile)

    print("BSTI train-test-val split: ",len(bsti_train_list), len(bsti_test_list), len(bsti_val_list))

    make_img_list(BSTI_TRAIN_FILE, bsti_train_list, BSTI_DATA_PATH)
    make_img_list(BSTI_VAL_FILE, bsti_val_list, BSTI_DATA_PATH)
    make_img_list(BSTI_TEST_FILE, bsti_test_list, BSTI_DATA_PATH)
    
#Include IIT-KGP Action Group Dataset
if args.kgp_action :
    # Construct the split lists
    kgp_train_list = []
    kgp_test_list = []
    kgp_val_list = []
    subdirs = os.listdir(KGP_DATA_PATH)
    
    for subdir in subdirs:
        subpath = os.path.join(KGP_DATA_PATH,subdir)
        kgp_images = [f for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))]
     
        for imgfile in kgp_images:
            rand_val = np.random.rand(1)
            if rand_val < 0.1:
                kgp_val_list.append(subdir+"/"+imgfile)
            elif rand_val < 0.3:
                kgp_test_list.append(subdir+"/"+imgfile)
            else:
                kgp_train_list.append(subdir+"/"+imgfile)

    print("KGP Action train-test-val split: ",len(kgp_train_list), len(kgp_test_list), len(kgp_val_list))

    make_img_list(KGP_TRAIN_FILE, kgp_train_list, KGP_DATA_PATH)
    make_img_list(KGP_VAL_FILE, kgp_val_list, KGP_DATA_PATH)
    make_img_list(KGP_TEST_FILE, kgp_test_list, KGP_DATA_PATH)