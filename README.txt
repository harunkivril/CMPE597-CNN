The scripts uses relative paths therefore please run them from the directory that contains the codes.

INSTALLATION:
=============
Required non default packages for python:
pytorch
torchvision
numpy
sklearn
matplotlib
tqdm

Also it is possible to create a conda environment with the environment.yml usign the following commands:

conda env create -f environment.yml
conda activate assignment

Using the conda environment also makes you able to use the same version with the author. 
If you are don't have conda you may want to install the compact version from the following link:
https://docs.conda.io/en/latest/miniconda.html

If you have gpu please consider changing "- pytorch=1.8.0" to "- pytorch-gpu=1.8.0" since it makes everything faster.

USAGE:
======
When "train.py" is run it automaticaly downloads data to ./data folder and trains a model with the parameters defined at the begining of of the script
It saves the state dict of best model to path defined as save_folder.
Save folder also will contain loss plot, accuracy plot, model object, 20th epoch state dict.
`python train.py`
or
`python3 train.py`
commands is enough to start it

When "eval.py" is called with similar python command, it checks the model_folder defined in the script and loads the model. After that it evaluates the data in test set and reports the test , validation and test loss with accuracy.
If the plot_latent_space is True after reporting the performance for every test image it gets the latent space of begining model, 20th epoch model and best model. Then it maps these latent  spaces to 2D by using TSN-e and saves the plots to model folder.

"model.py" contains the model class.
