# Image Classification Project

This project is a comprehensive pipeline for image classification tasks using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. It includes modules for data preparation, model creation, training, evaluation, and performance visualization.

## Table of Contents

- [Image Classification Project](#image-classification-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Model Creation](#model-creation)
    - [Training](#training)
    - [Evaluation and Visualization](#evaluation-and-visualization)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/EkmalRey/ImgClassification.git
   cd ImgClassification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

[Link To Colab](https://colab.research.google.com/drive/1Ma7x3H1a1lBH4H7KUipfrFL4kykCc2O4)

### Data Preparation

The `DataPreparation` class handles loading, processing, and splitting the dataset.

```python
from CNNPipeline import DataPreparation

dataset_folder = 'path/to/dataset'
data_prep = DataPreparation(dataset_folder)

# Load dataset
data = data_prep.load_dataset(resize=True, check_error=True, copy_to_local=True)

# Describe dataset
data_prep.describe_dataset(data)

# Split dataset
train_df, val_df, test_df = data_prep.split_dataset(data, train_size=0.8, upsample=True)

# Create image generators
train_gen, val_gen, test_gen = data_prep.image_generator(train_df, val_df, test_df, aug_train=True)

# Show sample images
data_prep.show_sample(train_gen)
```

### Model Creation

The `CNNModel` class provides functionalities to create and modify a CNN model.

```python
from CNNPipeline import CNNModel

model_folder = 'path/to/model_folder'
cnn_model = CNNModel(model_folder, classes=data_prep.classes)

# Create base model
base_model = cnn_model.create_basemodel_keras(base='EffNet', lr=0.001, trainable=False)

# Add layers to the base model
model_with_layers = cnn_model.add_fully_connected_layers(base_model, layer_units=[512, 256])
model_with_layers = cnn_model.add_batch_normalization(model_with_layers)
model_with_layers = cnn_model.add_dropout(model_with_layers, rate=0.4)
model_with_layers = cnn_model.add_flatten(model_with_layers)

# Compile the model
final_model = cnn_model.model_compile(model_with_layers)
```

### Training

Train the model using the created generators.

```python
# Train the model
history = cnn_model.train(final_model, train_gen, val_gen, epochs=10)

# Visualize training history
cnn_model.visualize_train(history)

# Export the trained model
cnn_model.export_model(final_model)
```

### Evaluation and Visualization

Evaluate the model on the test set and visualize the results.

```python
# Load the trained model
model_path = 'path/to/saved_model/model_best.keras'
loaded_model = cnn_model.import_model(model_path)

# Evaluate the model
cnn_model.evaluate(loaded_model, test_gen)

# Visualize performance
cnn_model.performance(k=1)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to include detailed information about the changes you made.

## License

This project is licensed under the Non-Commercial MIT License - see the [LICENSE](LICENSE) file for details.

---
