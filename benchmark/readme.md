# Test Script train_and_classify.py

This is a test script using the Foxglovetree dataset.
It allows you to check whether CUDA or MPS is available and measure how long the processing takes.

## Usage

Save `headshot_data.npy` and `headshot_labels.npy` in the `data` directory prepared in the current directory,
then run this script.

You can specify `cuda`, `mps`, or `cpu` as the first command-line argument.

The default is 50 epochs, but if you want to change the number of epochs,
specify a value such as `10` as the second command-line argument.
