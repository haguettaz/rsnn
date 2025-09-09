# Recurrent Spiking Neural Networks

*"Et tout d'un coup le souvenir m'est apparu."* - Marcel Proust, *Du côté de chez Swann*

### :boom: Spiking Neural Network

We consider recurrent neural networks comprising $L$ spiking neurons.
Each neuron continuously processes $K$ incoming signals and outputs a spike train in return.

### :book: Storing Spike Trains

Given one (or many) $L$-channels spike trains to store (and to reproduce later on), we turn the weights' learning task into a convex optimization problem.
The storage capacity (in duration of memories) per synapse appears to be non-vanishing.

### :link: Associative Recall

Despite working in continuous time, without a clock, and with imprecise and noisy neurons, our networks can store and recall prescribed spike trains (i.e., memories) with high temporal stability.
To the best of our knowledge, we are the first to explicitly demonstrate associative recall of memorized spike trains in continuous time.

## Installation

1. Optionally, create and activate a virtual environment.
    ```sh
    python -m venv rsnn
    source rsnn/bin/activate
    ```
    or 
    ```sh
    conda create -n rsnn
    conda activate rsnn
    ```

2. Clone this repository.
    ```sh
    git clone https://github.com/haguettaz/RSNN.git
    ```

3. Install the RSNN package and its dependencies.
    ```sh
    python -m pip install -e RSNN
    ```

## How to Use

<!-- :hourglass_flowing_sand: Work in progress... -->
A tutorial to start using the package is accessible [here](notebooks/how-to.ipynb).
<!-- It also gives a brief tour of the main results in [1]. -->

## Publications

### :hourglass_flowing_sand: Coming Soon ...
- H. Aguettaz and H.-A. Loeliger, *"Continuous-time neural networks can stably memorize random spike trains"*, arXiv, 2024.

### Prior Works
- P. Murer, *A New Perspective on Memorization in Recurrent Networks of Spiking Neurons*. Ph.D. dissertation, No. 28166, ETH Zürich, 2022.

