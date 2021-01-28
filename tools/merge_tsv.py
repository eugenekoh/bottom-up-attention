import csv
import sys
from tqdm import tqdm
from pathlib import Path
import logging
from collections import defaultdict
import pickle

csv.field_size_limit(sys.maxsize)

features_outfile = './data/tsv/merged.features.tsv'
labels_outfile = './data/tsv/merged.labels.tsv'
data_path = './data/features'

features_files = list(Path(data_path).glob('features.tsv.*'))
labels_files = list(Path(data_path).glob('labels.tsv.*'))

with open('./tools/cleaned_ids.pkl', 'rb') as f:
    cleaned_ids = pickle.load(f)

labels = {}
tqdm.write('retrieving labels')
with tqdm(total=210000) as pbar:
    for infile in labels_files:
        with open(infile) as tsv_in_file:
            reader = csv.reader(tsv_in_file, delimiter='\t')
            for item in reader:
                if item[0] in cleaned_ids:
                    labels[item[0]] = item
                pbar.update(1)

tqdm.write('writing out features')
with tqdm(total=len(cleaned_ids)) as pbar:
    with open(features_outfile, 'w') as features_tsv, open(labels_outfile, 'w') as labels_tsv:
        features_writer = csv.writer(features_tsv, delimiter = '\t')   
        labels_writer = csv.writer(labels_tsv, delimiter = '\t')  
        
        dummy_writer = csv.writer(open("/dev/null", 'w'), delimiter = '\t')
        
        for infile in features_files:
            with open(infile) as tsv_in_file:
                reader = csv.reader(tsv_in_file, delimiter='\t')
                for item in reader:
                    image_id = item[0]
                    if image_id not in cleaned_ids:
                        continue
                    try:
                        dummy_writer.writerow(item)
                        dummy_writer.writerow(labels[image_id])
                    except Exception as e:
                        tqdm.write(f'error for {image_id}, {str(e)}')
                        continue
                    
                    features_writer.writerow(item)
                    labels_writer.writerow(labels[image_id])
                    assert image_id == labels[image_id][0]
                    pbar.update(1)