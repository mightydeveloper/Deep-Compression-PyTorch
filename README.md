# Deep-Compression-PyTorch
PyTorch implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding'  by Song Han, Huizi Mao, William J. Dally

This implementation implements three core methods in the paper - Deep Compression
- Pruning
- Weight sharing
- Huffman Encoding

## Requirements
Following packages are required for this project
- Python3.6+
- tqdm
- numpy
- pytorch, torchvision
- scipy
- scikit-learn

or just use docker
``` bash
$ docker pull tonyapplekim/deepcompressionpytorch
```

## Usage
### Pruning
``` bash
$ python pruning.py
```
This command
- trains LeNet-300-100 model with MNIST dataset
- prunes weight values that has low absolute value
- retrains the model with MNIST dataset
- prints out non-zero statistics for each weights in the layer

You can control other values such as
- random seed
- epochs
- sensitivity
- batch size
- learning rate
- and others
For more, type `python pruning.py --help`

### Weight sharing
``` bash
$ python weight_share.py saves/model_after_retraining.ptmodel
```
This command
* Applies K-means clustering algorithm for the data portion of CSC or CSR matrix representation for each weight
* Then, every non-zero weight is now clustered into (2**bits) groups.
(Default is 32 groups - using 5 bits)
- This modified model is saved to
`saves/model_after_weight_sharing.ptmodel`

### Huffman coding
``` bash
$ python huffman_encode.py saves/model_after_weight_sharing.ptmodel
```
This command
- Applies Huffman coding algorithm for each of the weights in the network
- Saves each weight to `encodings/` folder
- Prints statistics for improvement



## Note
Note that I didn’t apply pruning nor weight sharing nor Huffman coding  for bias values. Maybe it’s better if I apply those to the biases as well, I haven’t try this out yet.

Note that this work was done when I was employed at http://nota.ai

