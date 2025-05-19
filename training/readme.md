Training Scripts Overview

This folder contains Python scripts for fine-tuning YOLOv8 and YOLOv9 models on real and synthetic data from the TTA video dataset. We provide separate configurations for two versions of the controlled real dataset:

-One with ~12,000 annotated frames (7 classes); and another with ~4,800 annotated frames (6 classes)

Then the idea is to finetune the models pretrained on controlled real data, using synthetic data. 

The goal is to evaluate how the amount of real-world supervision affects model performance before and after incorporating synthetic data for sim-to-real transfer.

 Dataset Versions and Classes
 
✅Controlled real data: 12K Sample Set (7 classes)
Used for full assembly monitoring, this dataset includes all components and their assembly states:
Tidal-turbine  
Body-assembled  
Body-not-assembled  
Hub-assembled  
Hub-not-assembled  
Rear-cap-assembled  
Rear-cap-not-assembled 

✅Controlled real data: 4800 Sample Set (6 classes)
A reduced version:
Tidal_Turbine
Body_Assembled
Body_Not_Assembled
Hub_Assembled
Hub_Not_Assembled
Rear_Cap

✅ Synthetic Data: 4800 Sample Set (7 classes)
Auto-labeled images generated using Unity’s Perception Package, following the same class structure as the 12k sample set:
Body-assembled  
Body-not-assembled  
Hub-assembled  
Hub-not-assembled  
Rear-cap-assembled  
Rear-cap-not-assembled 
Tidal-turbine 
These images are used for ablation studies and domain adaptation experiments.

🛠 Available Scripts

train_yolo_12k.py => Fine-tunes YOLOv8/v9 on the 12k-sample dataset with 7 classes
train_yolo_4800.py => Trains YOLOv8/v9 on the smaller 4800-sample dataset with 6 classes
train_synthetic.py => Continues training from a real-data-pretrained model using synthetic images from
