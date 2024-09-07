# Multi-Stage Two-Tower Recommender
A multi-stage movie recommendation system using YouTube's Tow-Tower architecture.

## Getting Started
### Installation
To set up the virtual environment and install the necessary dependencies, run the following command:
```bash
make install
```
This will ensure your environment is prepared with all required packages, ready to run the project.

### Usage
Run the [test.py](test.py) for a quick start.

## Dataset
Different versions of the [MovieLens](https://grouplens.org/datasets/movielens/) dataset are used for both training and evaluation purposes. Due to significant data redundancy in the original datasets, I created the script [dataset.py](scripts/dataset.py) to generate separate files for movies, users, and ratings, effectively eliminating duplication and reducing the overal size of each datasets along side feature engineering and feature selection for the task. This is most similar to the data at hand in a production environment.

### Feature Selection
- Retained all features from the original dataset, except for `raw_user_age`, which is only available in the `100k` version of the dataset.
- Removed `user_occupation_text` since `user_occupation_label` was derived from it, making it redundant.

### Feature Engineering
- Most of the `movie_title` values in the dataset include their release years in parentheses, which were extracted and treated as a separate feature. For movies without a listed release year, the corresponding values are left as `NaN`.
- Converted the `user_gender` feature from boolean values to integer representations.

## Project Structure
- [data/](data) - Directory for storing *100k* and *1m* versions of the [Movielense](https://grouplens.org/datasets/movielens/) dataset.
- [src/](src) - Contains the core implementation, including model definitions, architecture, and api functions.
- [scripts/](scripts) - Useful scripts.

## 🖇References
**Videos**
- [Building recommendation systems with TensorFlow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz)

**Papers**
- [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf)

**Libraries**
- [Tensorflow Ranking (TFR)](https://github.com/tensorflow/ranking)
- [TensorFlow Recommenders (TFRS)](https://github.com/tensorflow/recommenders)

**Codes**
- [xei/recommender-system-tutorial](https://github.com/xei/recommender-system-tutorial)

## Contributions
Contributions are welcome and greatly appreciated! If you have an idea for improvement, or if you find a bug, feel free to contribute by opening an [issue](https://github.com/keivanipchihagh/multi-stage-two-tower-recommender/issues) or a [pull request](https://github.com/keivanipchihagh/multi-stage-two-tower-recommender/pulls).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.