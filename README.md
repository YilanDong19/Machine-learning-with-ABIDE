# Machine learning with ABIDE (Google Colab)

##   Instruction: 

**1. Download First_step_prepare.ipynb (this file prepares everything)**

**2. Upload First_step_prepare.ipynb to google drive, open with google colab**

**3. Run First_step_prepare.ipynb**

The functions of First_step_prepare.ipynb :
1. Create folders
2. Download fMRI and sMRI data to google drive
3. Download necessary scripts

I also prepared a copy of all the necessary scripts in folder "scripts" for those who want to have a quick look. For those who want to run these scripts, you still need to run First_step_prepare.ipynb at the beginning.


### Models: 
SVM, FCN, AE-FCN, GCN and EV-GCN

### Dataset: 
ABIDE 870 samples, one subject failed with FreeSurfer pipeline

### Feature types: 
1. fMRI (AAL, CC200)
2. sMRI (our Freesurfer outputs, 7.0)
3. fMRI(CC200) + sMRI
               
### Ensemble methods: 
1. Max voting
2. EMMA
                  
### Harmonization method: 
- ComBat (https://github.com/Jfortin1/ComBatHarmonization)


##   Folder "scripts": 
I uploaded all the necessary scripts to this folder as a copy for those who want to have a quick look. Most of the scripts are named by "machine learning model" + "feature type", for example AUTO_fMRI.ipynb is to train Auto-encoder + fully connected neural network (AE-FCN) on fMRI features. The scripts named "EMMA_*" can only be run after all the other scripts, since EMMA ensemble method is to combine the outputs of all the machine learning models. 

