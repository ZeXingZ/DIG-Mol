<h1 style="border-bottom: 2px solid lightgray;">DIG-Mol: A Contrastive Dual-Interaction Graph Neural Network for Molecular Property Prediction</h1>

__Abstract:__ Molecular property prediction is a key component of AI-driven drug discovery and molecular characterization learning. Despite recent advances, existing methods still face challenges such as limited ability to generalize, and inadequate representation of learning from unlabeled data, especially for tasks specific to molecular structures. To address these limitations, we introduce DIG-Mol, a novel self-supervised graph neural network framework for molecular property prediction. This architecture leverages the power of contrast learning with dual interaction mechanisms and unique molecular graph enhancement strategies. DIG-Mol integrates a momentum distillation network with two interconnected networks to efficiently improve molecular characterization. The framework’s ability to extract key information about molecular structure and higherorder semantics is supported by minimizing loss of contrast. We have established DIG-Mol’s state-of-the-art performance through extensive experimental evaluation in a variety of molecular property prediction tasks. In addition to demonstrating superior transferability in a small number of learning scenarios, our visualizations highlight DIGMol’s enhanced interpretability and representation capabilities. These findings confirm the effectiveness of our approach in overcoming challenges faced by traditional methods and mark a significant advance in molecular property prediction. The code for this project is now available at https://github.com/ZeXingZ/DIG-Mol.

__Article Links:__ (arxiv) [https://arxiv.org/pdf/2405.02628] Accepted by IEEE-JBHI!

![](Fig.1.png)

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name DIG-Mol python=3.7
$ conda activate DIG-Mol

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of DIG-Mol
$ git clone https://github.com/ZeXingZ/DIG-Mol.git
$ cd DIG-Mol
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Data preprocessing
To convert SMILES strings into molecular graphs using RDKit, refer to the data processing code available in `dataset/dataset.py`.

RDKit link[https://github.com/rdkit/rdkit]

### Pre-training

To train the DIG-Mol, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python DIG-Mol.py
```

### Fine-tuning 

To fine-tune the DIG-Mol pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python DIG-Mol_finetune.py
```

### Pre-trained models

We also provide pre-trained DIGNN models, which can be found in `model.pth` and `model_50.pth` for different pretraining epoches respectively. 

## Acknowledgement

- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)
