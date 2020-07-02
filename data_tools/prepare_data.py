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
parser.add_argument("--bsti", action='store_true', default=False)
parser.add_argument("--kgp_action", action='store_true', default=False)
parser.add_argument("--aiims", action='store_true', default=False)
args = parser.parse_args()

COVID19_DATA_PATH = "./data/covid19"
COVID19_IMGS_PATH = "/home/cse/dual/cs5150296/scratch/COVID_Data/covid-chestxray-dataset"
BSTI_IMGS_PATH = "/home/cse/dual/cs5150296/scratch/COVID_Data/BSTI"
KGP_IMGS_PATH='/home/cse/dual/cs5150296/scratch/COVID_Data/IITKGP-Action/images/'
PNEUMONIDA_DATA_PATH = "/home/cse/dual/cs5150296/scratch/COVID_Data/chest-xray-pneumonia"
AIIMS_DATA_PATH = "/home/cse/dual/cs5150296/scratch/COVID_Data/AIIMS_complete_dataset"
DATA_PATH = "./data"

# Assert that the data directories are present
check_list = [COVID19_DATA_PATH, COVID19_IMGS_PATH, PNEUMONIDA_DATA_PATH, DATA_PATH]
if args.bsti:
    check_list.append(BSTI_IMGS_PATH)
if args.kgp_action:
    check_list.append(KGP_IMGS_PATH)
for d in check_list:
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
                
    # Prepare list using BSTI covid dataset
    if args.bsti:
        bsti_covid_file = os.path.join(COVID19_DATA_PATH, 'bsti_%s_list.txt'%split)
        with open(bsti_covid_file, 'r') as cf:
            for f in cf.readlines():
                f = os.path.join(BSTI_IMGS_PATH, f.strip())
                if args.combine_pneumonia:
                    l.append((f, 2)) # Class 2
                else:
                    l.append((f, 3)) # Class 3
    
    # Prepare list using IIT-KGP Action Group covid dataset
    if args.kgp_action:
        kgp_covid_file = os.path.join(COVID19_DATA_PATH, 'kgp_%s_list.txt'%split)
        with open(kgp_covid_file, 'r') as cf:
            for f in cf.readlines():
                f = os.path.join(KGP_IMGS_PATH, f.strip())
                if args.combine_pneumonia:
                    l.append((f, 2)) # Class 2
                else:
                    l.append((f, 3)) # Class 3

    with open(os.path.join(DATA_PATH, '%s.txt'%split), 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)
            
def create_test_list (split):
    assert split in ['test']

    l = []
    # Prepare list using AIIMS dataset
    for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Normal', '*')):
        l.append((f, 0)) # Class 0

    for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Pneumonia', '*')):
        if args.combine_pneumonia:
            l.append((f, 1)) # Class 1
        else:
            if 'bacteria' in f:
                l.append((f, 1)) # Class 1
            else:
                l.append((f, 2)) # Class 2
        
                
    for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Covid', '*')):
        if args.combine_pneumonia:
            l.append((f, 2)) # Class 2
        else:
            l.append((f, 3)) # Class 3

    with open(os.path.join(DATA_PATH, 'aiims_%s.txt'%split), 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)

if args.aiims:
    create_test_list('test')
else:
    for split in ['train', 'test', 'val']:
        create_list(split)