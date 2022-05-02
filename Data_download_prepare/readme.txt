############
Attention!!!  Attention!!!  Attention!!!
Please use nilearn==0.6.2, otherwise the database cannot be downloaded successfully！！！！！
############
The aim of this folder is to download the ABIDE dataset, and then create the 5-fold cross validation dataset for SVM, FCN, Auto-encoder and GCN models.
To run:
1. Add path in the fetch_data.py and ABIDEParser.py
2. Run fetch_data.py

!!!!! Please use nilearn==0.6.2, otherwise the database cannot be downloaded successfully ！！！！！

Output:Three folders (in the download_data folder, which will appear after running the fetch_data.py)
1. all : contains 871 function connectiontivity matrices
2. cross validation : 5-fold cross validaiton dataset
3. measures : the labels, ages, genders and collection sites .mat files of 871 samples

These three folders will be used in SVM, FCN, Auto-encoder and GCN experiments.
