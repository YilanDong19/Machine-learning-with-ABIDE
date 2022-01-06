from nilearn import datasets
import ABIDEParser as Reader
import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio

# Selected pipeline
pipeline = 'cpac'

# Input data variables
num_subjects = 871  # Number of subjects
root_folder = 'Input your own path (example: D:\Data_download_prepare   should be the same with the root_folder in fetch_data.py)'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
new_data_folder = os.path.join(root_folder, 'download_data\\all')
cross_validation_path = os.path.join(root_folder, 'download_data\\cross validation')
measures_path = os.path.join(root_folder, 'download_data\\measures')

# Files to fetch
files = ['rois_aal']

filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_aal': 'rois_aal.1D'}

if not os.path.exists(data_folder): os.makedirs(data_folder)
if not os.path.exists(new_data_folder): os.makedirs(new_data_folder)
if not os.path.exists(cross_validation_path): os.makedirs(cross_validation_path)
if not os.path.exists(measures_path): os.makedirs(measures_path)
shutil.copyfile('./subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))

# Download database files
abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                                 band_pass_filtering=True, global_signal_regression=False, derivatives=files)


subject_IDs = Reader.get_ids(num_subjects)
subject_IDs = subject_IDs.tolist()

# Create a folder for each subject
for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0])):
    subject_folder = os.path.join(data_folder, s)
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Get the base filename for each subject
    base = fname.split(files[0])[0]

    # Move each subject file to the subject folder
    for fl in files:
        if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
            shutil.move(base + filemapping[fl], subject_folder)

time_series = Reader.get_timeseries(subject_IDs, 'aal')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
        Reader.subject_connectivity(time_series[i], str(i), 'aal', 'correlation')

#################################### prepare dataset with 5 cross validation ############################################################
labels_dist = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
sites_dist = Reader.get_subject_score(subject_IDs, score='SITE_ID')
labels = list(labels_dist.values())
sites = list(sites_dist.values())
unique_sites = np.unique(list(sites_dist.values())).tolist()
k_fold = 5
################## create folders
if os.path.exists(cross_validation_path):
    shutil.rmtree(cross_validation_path)

os.mkdir(cross_validation_path)


for i in range(k_fold):
    os.mkdir(cross_validation_path + '\\' + 'group' + str(i))
    os.mkdir(cross_validation_path + '\\' + 'group' + str(i) + '\\' + 'train')
    os.mkdir(cross_validation_path + '\\' + 'group' + str(i) + '\\' + 'validation')
    os.mkdir(cross_validation_path + '\\' + 'group' + str(i) + '\\' + 'test')

for each_site in unique_sites:
    index_site = Reader.get_index(sites, each_site)
    label = np.zeros((len(index_site)))
    for i in range(len(index_site)):
        index = index_site[i]
        label[i] = int(labels[int(index)])
    ############## StratifiedKFold
    sfolder = StratifiedKFold(n_splits=k_fold,random_state=0,shuffle=True)
    group = 0
    for train, validation in sfolder.split(index_site,label):
        for i in train:
            name = index_site[i]
            oldname = new_data_folder + '\\' + str(name) + '.mat'
            newname = cross_validation_path + '\\' + 'group' + str(group) + '\\' + 'train' + '\\' + str(name) + '.mat'
            shutil.copyfile(oldname, newname)
        for j in validation:
            name = index_site[j]
            oldname = new_data_folder + '\\' + str(name) + '.mat'
            newname = cross_validation_path + '\\' + 'group' + str(group) + '\\' + 'validation' + '\\' + str(name) + '.mat'
            shutil.copyfile(oldname, newname)
        group = group+1

    group = 0
    for train, validation in sfolder.split(index_site,label):
        if group == 0:
            for j in validation:
                name = index_site[j]
                oldname = cross_validation_path + '\\' + 'group' + str(group + k_fold - 1) + '\\' + 'train' + '\\' + str(name) + '.mat'
                newname = cross_validation_path + '\\' + 'group' + str(group + k_fold - 1) + '\\' + 'test' + '\\' + str(name) + '.mat'
                shutil.move(oldname, newname)
        else:
            for j in validation:
                name = index_site[j]
                oldname = cross_validation_path + '\\' + 'group' + str(group - 1) + '\\' + 'train' + '\\' + str(name) + '.mat'
                newname = cross_validation_path + '\\' + 'group' + str(group - 1) + '\\' + 'test' + '\\' + str(name) + '.mat'
                shutil.move(oldname, newname)
        group = group+1


################### save labels, collection sites, age, gender, FIQS .mat files #################################
ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
genders = Reader.get_subject_score(subject_IDs, score='SEX')
labels_array = np.zeros((2,num_subjects))
ages_array = np.zeros((num_subjects))
genders_array = np.zeros((num_subjects))

for i in range(num_subjects):
    if labels[i] == '1':
        labels_array[0, i] = 1
    else:
        labels_array[1, i] = 1

sio.savemat(os.path.join(measures_path, 'ages.mat'), {'ages': list(ages.values())})
sio.savemat(os.path.join(measures_path, 'genders.mat'), {'genders': list(genders.values())})
sio.savemat(os.path.join(measures_path, 'sites.mat'), {'sites': sites})
sio.savemat(os.path.join(measures_path, 'ABIDE_label_871.mat'), {'label': labels_array})