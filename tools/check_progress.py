import csv
import sys
import tqdm
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'images']
LABELNAMES = ['image_id', 'list']
found_ids = set()
features_file = 'data/features/yfcc.tsv.0'
with open(features_file) as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
    with tqdm.tqdm(total=220000) as pbar:
        for x in reader:
            if x['image_id'] in found_ids:
                tqdm.write(x['image_id'])
            found_ids.add(x['image_id'])
            pbar.update(1)
