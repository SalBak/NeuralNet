# Neural Network for Adulteration Prediction

This project implements a customizable feedforward neural network using NumPy and pandas to predict adulteration levels in a dataset.
The network topology, activation functions, and optimization parameters are defined externally via a configuration file.

## Features

- Manual data preprocessing with missing value handling
- Flexible architecture via external topology configuration
- Multiple activation functions: ReLU, GELU, Leaky ReLU, Sigmoid, Tanh, Linear
- Support for MSE and Cross-Entropy loss functions
- Adam optimizer with configurable parameters
- One-hot encoding for multiclass classification ([0, 2.5, 5])
- Binarized output using the maximum class probability
- Training with mini-batch gradient descent
- Custom accuracy calculation on test data

## Files

- `NeuralNet.py`: Main script containing data handling, model architecture, training, and inference
- `topology.txt`: Configuration file specifying neural network structure and training parameters
- `ADULTERED_TRAIN.csv`: Training dataset (not provided)
- `ADULTERED_TEST_NOCLASSES.csv`: Test dataset (not provided)

## Requirements

- Python 3.6+
- NumPy
- pandas
- seaborn

Install dependencies:

```bash
pip install numpy pandas seaborn
```

## Topology Configuration (`topology.txt`)

```
hidden_layers=3
hidden_nodes=40,16,6
activation_functions=sigmoid,sigmoid,linear
learning_rate=0.001
beta1=0.9
beta2=0.999
epsilon=1e-8
loss_function=cross_entropy
```

## Usage

1. Ensure `ADULTERED_TRAIN.csv` and `ADULTERED_TEST_NOCLASSES.csv` are in the working directory.
2. Update `topology.txt` to match desired architecture.
3. Run the training:

```bash
python NeuralNet.py
```

4. The model will train and output loss and accuracy every 10 epochs.
5. After training, predictions on the test set will be printed.

## Notes

- The network input expects 20 features.
- Final output layer uses 3 nodes, corresponding to class labels [0, 2.5, 5].
- Training is performed for 1000 epochs with batch size 32 by default.
