
# VSGenerator

**DVSNet: Dynamic Virtual Space Generation Neural Network**

[![PyPI Version](https://img.shields.io/pypi/v/VSGenerator.svg)](https://pypi.org/project/VSGenerator/)

## Overview

VSGenerator provides `DVSNet`, a neural network designed to dynamically generate virtual spaces for training datasets. This project is an extension of the [Bgolearn](https://github.com/Bin-Cao/Bgolearn) platform and focuses on applying neural networks to material design problems. 

The primary goal of `DVSNet` is to enhance the generation of training data using the SMOGN algorithm, which introduces virtual data points based on noise and neighbor analysis, improving model robustness and performance in material informatics applications.

## Installation

You can install the package using pip:

```bash
pip install VSGenerator
```


## Usage

To train a model using `DVSNet`, you can configure the necessary parameters as follows:

```python
from VSGenerator import DVSNet

data_file = './dataset/traindata.csv'
input_dim = 4
y_dim = 1
latent_dim = 15
epochs = 500
batch_size = 16
patience = 20

# Train the DVSNet model
DVSNet.train(data_file, input_dim, y_dim, latent_dim, epochs=500, batch_size=32, patience=20)
```

### Parameters

- `data_file`: Path to the training data CSV file.
- `input_dim`: Number of features (input dimension).
- `y_dim`: Number of targets (output dimension).
- `latent_dim`: Dimension of latent variables in the encoding space.
- `epochs`: Number of training epochs.
- `batch_size`: Number of samples per batch.
- `patience`: Number of epochs to wait for improvement before early stopping.
- `SMOGN`: Whether to use the SMOGN algorithm for virtual data generation (`True` or `False`).
- `k_neighbors`: Number of neighbors to use in the SMOGN algorithm.
- `noise_factor`: Noise factor used in virtual data generation.

## Example

Here’s a complete example to get started with `DVSNet`:

```python
data_file = './dataset/traindata.csv'
input_dim = 4
y_dim = 1
latent_dim = 15
epochs = 500
batch_size = 16
patience = 20

DVSNet.train(
    data_file=data_file, 
    input_dim=input_dim, 
    y_dim=y_dim, 
    latent_dim=latent_dim, 
    epochs=epochs, 
    batch_size=batch_size, 
    patience=patience
)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you encounter any issues or have suggestions for improving the code, feel free to:

- Open an issue in the [GitHub repository](https://github.com/Bin-Cao/Bgolearn)
- Contact the author:
  - **Bin Cao** 
  - **Affiliation**: Guangzhou Municipal Key Laboratory of Materials Informatics, Advanced Materials Thrust, Hong Kong University of Science and Technology (Guangzhou), Guangdong, China
  - **Email**: bcao686@connect.hkust-gz.edu.cn

Thank you for choosing `VSGenerator` and `Bgolearn` for your material design research!
