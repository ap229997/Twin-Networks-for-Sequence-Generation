## Twin Networks for Sequential Pixel Generation
PyTorch implementation for ICLR 2018 accepted paper *Twin Networks: Matching the Future for Sequence Generation* [[pdf]](https://openreview.net/pdf?id=BydLzGb0Z) for verifying the results for pixel-by-pixel generation on MNIST.

### Setup
This repository is compatible with python 2. </br>
- Follow instructions outlined on [PyTorch Homepage](https://pytorch.org/) for installing PyTorch (Python2).

### Data
Hugo's binarized version of MNIST is used for this repo.
Both the MNIST and the binarized version can be downloaded by running the script ```get_mnist.sh```.

```load.py``` contains the functionality for loading and processing data into train, val and test split. It is internally called by training script for loading data.

### Train the model
The model can be trained for both unconditioned and conditioned generation. The respective commands are:
```
python train_seqmnist.py --num_epochs 10 --rnn_dim 1024 --bsz 64 --twin 0.0
python train_condmnist.py --num_epochs 10 --rnn_dim 1024 --bsz 64 --twin 0.0
```
In case of conditional generation, the label information is concatenated to the input and passed to the model. Please refer to ```train_seqmnist.py``` for implemetation details.

```--twin``` can be used to specify whether to use the twin network or not and what weight should be assigned to the twin loss component in the total loss. A value of 0.0 corresponds to the model without twin network whereas value > 0.0 corresponds to the model with the twin network.

Please refer to ```train_seqmnist.py``` for more information on different parameters which are to be used while training.

### Evaluation
Negative log likelihood is computed for both the val and test split of MNIST after each training epoch.

For pixel by pixel sequential generation, run the script ```generate_mnist.py```.

### Results
The results for Negative log likelihood (NLL) for both unconditioned and conditioned generation are specified in [Reproducibility_Report.pdf](https://github.com/ap229997/Twin-Networks-for-Sequence-Generation/blob/master/Reproducibility_Report.pdf).

In addition, training logs are provided in ```seqmnist_twin_logs``` and ```condmnist_twin_logs``` for few variations of different parameters which show the convergence of loss.
