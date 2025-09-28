# Genetic Algorithm based Hyperparameter Optimisation for Convolutional Neural Networks

Finding optimal hyperparameters for Convolutional Neural Networks (CNNs) is tedious and difficult, especially with non-IID data and a mix of continuous and discrete hyperparameters. Genetic Algorithms (GA) can efficiently explore such complex search spaces without using gradients.


## Chromosome Representation

The GA optimises CNN hyperparameters by representing each individual as a chromosome, which is an instance of the `Hyperparameters` dataclass. Each gene in the chromosome corresponds to a specific hyperparameter:

- `width_mult`: scales convolutional channels
- `learning_rate`: SGD learning rate
- `batch_size`: training sample count before weight update
- `dropout_rate`: dropout applied in the model
- `weight_decay`: L2 regularisation
- `momentum`: SGD momentum
- `conv_channels`: base channel counts _(kept constant)_

Default ranges for initialising chromosomes:

| Hyperparameter | Initial Range   |  Clipped Range   |
|----------------|-----------------|------------------|
| width_mult     | 0.5 – 1.5       | 0.25 – 2.0       |
| learning_rate  | 0.005 – 0.05    | 1e-4 – 0.1       |
| batch_size     | [64, 96, 128, 160, 192] | N/A      |
| dropout_rate   | 0.1 – 0.4       | 0.0 – 0.5        |
| weight_decay   | 1e-5 – 5e-3     | 1e-6 – 1e-2      |
| momentum       | 0.8 – 0.95      | 0.1 – 0.99       |


## How the Genetic Algorithm works

1. **Initial Population:** Starts with a baseline individual and N-1 random chromosomes.
2. **Evaluation:** Each individual is trained and tested, and its fitness score is calculated based on validation accuracy.
3. **Selection:** A tournament method is used to pick a pool of parents, giving higher chances to individuals with better scores.
4. **Crossover:** For every attribute, random swaps between two parents are performed to create new offspring with mixed traits.
5. **Mutation:** Small random changes are applied to each gene (using Gaussian noise or random choices), and values are clipped to stay within valid ranges.
6. **Elitism:** The best performing individual from the current generation is always preserved and carried into the next generation.

## Project Structure
```
src/
 ├── models.py          # Separable-conv CNN
 ├── dataset.py         # Small wrapper (FEMNISTDataset) for the Hugging Face FEMNIST split
 ├── genetic.py         # Hyperparameters dataclass and compact GA 
 ├── train.py           # Trainer to evaluate an individual's fitness
 ├── main.py            # Orchestrates dataset loading, GA loop, and final training/evaluation
 ├── predict.py         # Helper to load a saved model and classify images from a folder
 └── run_predict.py     # Script to run predictions using predict.py
```


## How to run

1. **Install dependencies:**
   
   It is recommended to use a virtual environment (`venv`). Install the required packages:

   ```bash
   python -m pip install torch torchvision datasets pillow tqdm
   ```

2. **Run `main.py`:**

   This starts the genetic algorithm search and trains the final model:

   ```bash
   python -m src.main
   ```

3. **Run the Prediction Helper:**
    
   Edit `src/run_predict.py` to set the path to your image test directory and the trained model file, then run:

   ```bash
   python -m src.run_predict
   ```

## License

This project is licensed under the [MIT LICENSE](LICENSE).