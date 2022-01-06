This is the application of the SVM model on ABIDE dataset.

To run:
1. The first step is important, you must run Data_download_prepare\\fetch_data.py before you run this code. fetch_data.py can download ABIDE dataset, 
create the 5-fold cross validation dataset. Please confirm which atlas you are using. At present, the default atlas in this code is cc200 (200*200), if you use other atlases, you need to adjust the image size in the code. fetch_data.py wil generate three folders: all, cross validation and measures. The pathes of these folder will be used
in the SVM_main.py
2. Fill in the pathes in the SVM_main.py (root_path = the path of all folder, cross_validation_path = the path of cross validation folder, 
label_dir = the path of measures folder, save_path = choose a path to save models, excel_root = a path where has a excel file to store the accuracy
loss during training, validation and testing).
3. Run SVM_main.py
4. The results will be recorded in a excel file under the excel_root path.
