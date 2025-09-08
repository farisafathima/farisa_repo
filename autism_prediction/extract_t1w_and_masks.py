import os
import shutil


source_dir = "/mnt/data/shyam/farisa/ASD_proj/data/fmriprep"
destination_dir = "/mnt/data/shyam/farisa/ASD_proj/data/extracted_data"


source_dir = os.path.expanduser(source_dir)
destination_dir = os.path.expanduser(destination_dir)


os.makedirs(destination_dir, exist_ok=True)

# Iterate through subject folders
for subject_folder in os.listdir(source_dir):
    subject_path = os.path.join(source_dir, subject_folder, "anat")
    if os.path.exists(subject_path) and os.path.isdir(subject_path):
        #print(f"Checking directory: {subject_path}")  
        for file_name in os.listdir(subject_path):
            if "space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz" in file_name or \
               "space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz" in file_name:
                file_path = os.path.join(subject_path, file_name)
                #print(f"Found file: {file_name} in {subject_path}")  

                
                dest_folder = os.path.join(destination_dir, subject_folder)
                os.makedirs(dest_folder, exist_ok=True)

                
                shutil.copy(file_path, dest_folder)
                #print(f"Copied {file_name} to {dest_folder}")  

print(f"Extracted files have been copied to {destination_dir}")
