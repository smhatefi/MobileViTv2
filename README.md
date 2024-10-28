# MobileViTv2
This repository contains the implementation of MobileViTv2 model in Keras 3.

## Project Structure
- `mobilevit_v2.py`: Contains the main model implementation.
- `configs.py`: Contains the model configurations.
- `test.py`: Script for evaluating the model.
- `utils/base_layers.py`: Contains base layers.
- `utils/linear_attention.py`: Contains Separable Self-attention implementation.
- `utils/mobilevit_v2_block.py`: Contains MobileViTv2 block implementation.
- `utils/utils.py`: Contains utility functions.

## Usage

For evaluating the model on an example images run the `test.py` script:
```
python test.py
```

This will:
   - Sets the Keras 3 backend.
   - Downloads an example image from the Web.
   - Makes three different MobileViTv2 models with width multipliers α=0.5,1,2 and downloads the pre-trained weights.
   - Test these three models on the example image.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The MobileViTv2 model architecture is inspired by the original [MobileViTv2 paper](https://arxiv.org/abs/2206.02680).
- The code is heavily borrowed from [this github repo](https://github.com/veb-101/keras-vision).
