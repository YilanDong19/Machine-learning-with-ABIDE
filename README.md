# Machine learning with ABIDE (Google Colab)

# Instruction: 

**1. Download First_step_prepare.ipynb (this file prepares everything)**

**2. Upload First_step_prepare.ipynb to google drive, open with google colab**

**3. Run First_step_prepare.ipynb**

The functions of First_step_prepare.ipynb :
1. Create folders
2. Download fMRI and sMRI data to google drive
3. Download necessary scripts

I also prepared a copy of all the necessary scripts in folder "scripts" for those who want to have a quick look. For those who want to run these scripts, you still need to run First_step_prepare.ipynb at the beginning.


### Models: 
1. SVM 
2. FCN
3. AE-FCN [1] 
4. GCN [2]
5. EV-GCN [3]

### Dataset: 
ABIDE 870 samples, one subject failed with FreeSurfer pipeline

### Feature types: 
1. fMRI (AAL, CC200)
2. sMRI (our Freesurfer outputs, 7.0)
3. fMRI(CC200) + sMRI
               
### Ensemble methods: 
1. Max voting [4]
2. EMMA [5]
                  

###   Folder "scripts": 
I uploaded all the necessary scripts to this folder as a copy for those who want to have a quick look. Most of the scripts are named by "machine learning model" + "feature type", for example AE_FCN_fMRI.ipynb is to train Auto-encoder + fully connected neural network (AE-FCN) on fMRI features. The scripts named "EMMA_*" can only be run after all the other scripts, since EMMA ensemble method is to combine the outputs of all the machine learning models. 

###  References
[1] Rakić, Mladen, Mariano Cabezas, Kaisar Kushibar, Arnau Oliver, and Xavier Lladó. 2020. ‘Improving the Detection of Autism Spectrum Disorder by Combining Structural and Functional MRI Information’. NeuroImage: Clinical 25(November 2019). doi: 10.1016/j.nicl.2020.102181. 

[2] Parisot, Sarah, Sofia Ira Ktena, Enzo Ferrante, Matthew Lee, Ricardo Guerrero, Ben Glocker, and Daniel Rueckert. 2018. ‘Disease Prediction Using Graph Convolutional Networks: Application to Autism Spectrum Disorder and Alzheimer’s Disease’. Medical Image Analysis 48:117–30. doi: 10.1016/j.media.2018.06.001. 

[3] Huang, Yongxiang, and Albert C. S. Chung. 2020. ‘Edge-Variational Graph Convolutional Networks for Uncertainty-Aware Disease Prediction’. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 12267 LNCS:562–72. doi: 10.1007/978-3-030-59728-3_55. 

[4] Thomas G. Dietterich. 2000. ‘Ensemble Methods in Machine Learning’. Multiple Classifier Systems. 

[5] Kamnitsas, Konstantinos, Wenjia Bai, Enzo Ferrante, Steven McDonagh, Matthew Sinclair, Nick Pawlowski, Martin Rajchl, Matthew Lee, Bernhard Kainz, Daniel Rueckert, and Ben Glocker. 2017. ‘Ensembles of Multiple Models and Architectures for Robust Brain Tumour Segmentation’.

[6] Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. Harmonization Of Multi-Site Diffusion Tensor Imaging Data. NeuroImage, 161, 149-170, 2017
