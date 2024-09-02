# two-tower-recommender
A Movie Recommender System using YouTube's Two-Tower Architecture

## Getting Started
### Installation
To set up the virtual environment and install the necessary dependencies, run the following command:
```bash
make install
```
This will ensure your environment is prepared with all required packages, ready to run the project.

### Usage
...
Once the environment is set up, you can start using the recommender system for tasks like training, evaluation, or inference.
- **Training the Model:**
- **Evaluating the Model:**
- **Inference:**

## Dataset
The primary dataset used in this project is [Movielense](https://grouplens.org/datasets/movielens/). The data is stored in the [data/](data) directory and can be automatically downloaded or linked through scripts.

## Project Structure
- [data/](data) - Directory for storing [Movielense](https://grouplens.org/datasets/movielens/) datasets (*100k* and *1m* versions) as parquet files.
- [src/](src) - Contains the core implementation, including model definitions, architecture, and utility functions.
- [scripts/](scripts) - Utility scripts.


## References
- [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf)
- [Tensorflow Ranking (TFR)](https://github.com/tensorflow/ranking)
- [TensorFlow Recommenders (TFRS)](https://github.com/tensorflow/recommenders)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.