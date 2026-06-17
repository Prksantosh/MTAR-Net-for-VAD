# MTAR-Net-for-VAD
MTAR-Net: Memory-Guided Temporal Attention Recurrent Learning with Spatial Feature Enhancement for Video Anomaly Detection

This repository is under build and necessary source files are available for review purpose.

# Data availability:
The CUHK Avenue dataset is available at https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html.

ShanghiTech datasets are available at https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBampVcWlKWnNqOHdoTHQtMUFCZXJUVC05ZUg5QWc%5FZT1lSmJZNlk&id=303FB25922AAD438%2173214&cid=303FB25922AAD438 

UCSD Ped2 dataset is available at
http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm 
Dataset structure:

    Data/
    └── UCSD/
        ├── training_frames/
            └── Train001/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train002/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train004/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train005/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
        └── validation_frames/
                └── Train002/
                    └── 001.tif/
                    └── 002.tif/
                    └── 003.tif/
                    └── 004.tif/
        └── testing_frames/
                └── Test001/
                    └── 001.tif/
                    └── 002.tif/
                    └── 003.tif/
                    └── 004.tif/
# Usgae
    Set training and validation paths in train.py
    Run train.py to train the model.
    For evaluation, run test.py to get visualization maps, Area under RoC curve score, equal error rate score and anomaly detection curves.
    (Groundtruth will be requited to evaluate the model)
    
For any query, kindely contact santoshc@iiitm.ac.in
