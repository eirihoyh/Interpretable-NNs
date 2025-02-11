# Interpretable-NNs

This repository was made before the 39th international workshop on statistical modelling (IWSM) (https://iwsm2025.ie/). 

We looked into two classification problems: one linear, and one non-linear. Data is created using the create_data_unif function in the pipeline_functions.py file, under the folder islbbnn. 

Experiments can be found in the folders lrt_implementation and ann_implementation. Both folders contain all the trained networks, the results, and all active paths in the trained models (under path_graphs folder). We also included visualizations of local contributions and global explanations based on aggregated local contributions under the folder local_contributions. We recommend looking into the look_into_explanations.ipynb and the look_into_explanations_ann.ipynb files to both see how the explanations compare against each other (LBBNN vs. ANN and linear vs. non-linear), and to see how explanations might differ from those obtained from LIME and SHAP.  

All models are run with input-skip, where the input variables are concatenated to all hidden layers. The LBBNN model with LRT and the ANN model can be found in the networks folder under the islbbnn folder (lrt_net.py and ann_net.py, respectively). We will note that the models assumes that the bias term is added into the dataset before initialization.  


Abbreviations: 
* LBBNN: latent binary Bayesian neural network
* LRT: local reparameterization trick
* ANN: artificial neural network
