This folder contains the application of DNN and auto-encoder+MLP modelS to ABIDE dataset.

To run:
1. The first step is important, you must run Data_download_prepare\\fetch_data.py before you run this code. fetch_data.py can download ABIDE dataset, 
create the 5-fold cross validation dataset. It wil generate three folders: all, cross validation and measures. The pathes of these folder will be used
in the DNN_main.py and auto_encoder_MLP_main.py
2. Fill in the pathes in the DNN_main.py and auto_encoder_MLP_main.py (root_path = the path of all folder, cross_validation_path = the path of cross 
validation folder, label_dir = the path of measures folder, save_path = choose a path to save models, excel_root = a path where has a excel file to store the accuracy
loss during training, validation and testing)
3. Please confirm which atlas you are using. CC200 atlas: image_size = [200,200], AAL atlas: image_size = [116,116]
4. Run DNN_main.py or auto_encoder_MLP_main.py
5. The results will be recorded in a excel file under the excel_root path.
