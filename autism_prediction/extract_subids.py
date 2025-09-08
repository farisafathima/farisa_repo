
import os
import pandas as pd

dir_path = r'/mnt/data/shyam/farisa/ASD_proj/data/fmriprep'

data = []
for folder_name in os.listdir(dir_path):
    if folder_name.startswith('sub-') and not folder_name.endswith('.html'):
        sub_id = folder_name.split('sub-')[1][7:]
        target = 0 if 'control' in folder_name else 1
        data.append({'SUB-ID' : sub_id, 'TARGET' : target})


df = pd.DataFrame(data)
csv_path = r'/mnt/data/shyam/farisa/ASD_proj/data/subject_ids.csv'
df.to_csv(csv_path, index = False)
print("Done")
