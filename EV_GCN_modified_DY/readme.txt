This project is based on Huang's EV-GCN code, the original code is available here: https://github.com/SamitHuang/EV_GCN
In this project, we applied EV-GCN model to ABIDE dataset to have a performance comparison with other machine learning methods. To have the same evaluation standard, 
we modified the original EV-GCN code: 
1. We use 5-fold cross-validaiton(Huang: 10-fold cross-validation) 
2. Input all the connections in the population graph (Huang: connections in the upper triangle area)
3. Record the accuracy and loss during training, validaiton and testing

To run this project:
1. Add paths in fetch_data.py, ABIDEParser.py and 5cross_validation.py
2. Run fetch_data.py to download the ABIDE dataset
3. Run 5cross_validation.py
