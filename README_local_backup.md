# Lectin to Glycoprofile enhanced with Data-driven method
A cutting-edge machine-learning model that predicts glycan structures and abundance based on a lectin binding profile. Dominant lectins that are responsible for reconstructing the glycoprofile were revealed after using SHAP to interpret the model's behavior.

## Dependencies

### Python requirements:
The code was developed using Python 3.9.7.

### TensorFlow requirements:
The code was developed using tensorflow 2.11.0.

## Install LeGenD

### Clone from GitHub:
```
git clone https://github.com/Kiki988888/LeGenD/tree/main
cd LeGenD
```

### Set up an environment and install requirements
```
conda create -n LeGenD python=3.9.7 anaconda
conda activate LeGenD
conda env update --file environment.yml --name LeGenD
```
Or, you can install requirements using pip:
```
pip install -r requirements.txt
```

## Steps
The notebooks are numbered according to the order of steps. The data used for our work is stored under the [`Alpha`](Data/Alpha) folder.

[`0.Data_Preperation.ipynb`](Code/0.Data_Preperation.ipynb)

This file takes in our [glycoprofile dataset](Data/Voldborg_glycomics.xlsx) and [lectin-motif association p value table](Data/S4_Motif-associated%20p%20values.xlsx) from [Bojar et al](https://doi.org/10.1021/acschembio.1c00689). Preprocessing them to get ready-to-use for this study.

[`1.Simulate_LP.ipynb`](Code/1.Simulate_LP.ipynb)

This file simulates lectin profiles using glycoprofiles with the lectin binding rules. Then, it requires at least two samples of lectin profile and glycoprofile pairs to do a regression fitting, providing a robust congruence between the simulated and experimental lectin profiles. Training lectin profiles are then simulated in this file.

[`2.Model_TrainTest.ipynb`](Code/2.Model_TrainTest.ipynb)

This file trains and tests the model with the data we just curated. The predictions will be plotted as bar plots to demonstrate the model's performance.

[`3.SHAP.ipynb`](Code/3.SHAP.ipynb)

This file interprets the trained models with SHAP. SHAP values for certain glycans will be plotted to understand how the model made predictions.
