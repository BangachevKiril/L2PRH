Scripts for downloading the three main datasets used in the experiments
- Datasets are COCO, (a subset of) CC3M, (a subset of) Visual Genome, and the 50k most common words in English. All files are downloaded and stored in a common COCO format which makes subsequent processing easier. Provided you have installed the necessary environment GPUenv and change the output folders and cluster meta-data, you should be good to launch the slurm files.

- !CAUTION! on Visual Genome! The current python script downloads a few extra captions corresponding to empty images. You should run "visualgenome_find_corrupted.py" and then remove manually the extra captions.    
