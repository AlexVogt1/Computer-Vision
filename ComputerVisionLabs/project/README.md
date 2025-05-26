# Semantic Segmentation of Puzzle Peices using Gaussian Mixture Models and Deep CNN's

## Gaussian Mixture Model

The code for training and testing the gaussian mixture model can be found in `GaussianMixtureModel.ipynb`. It performs the EM algorithm until the parameters change less than some give tolerance $\epsilon$.

The training images should be placed in `./puzzle_corners_1024x768/images-1024x768/`, and masks in `./puzzle_corners_1024x768/masks-1024x768/`.

The hyperparameters are as follows:

```
$ gmms [int]: Number of clusters to use
$ kmeans [boolean]: Initialisation of cluster centers using Kmeans clustering
$ tol [float]: Tolerance for training completion
$ min_iter [int]: Minimum number of iterations before training can complete
$ max_iter [int]: Maximum number of iterations for training to occur 
```

Dependencies can be installed as follows:

`python3 -m pip install natsort matplotlib glob numpy imagio scikit-image scipy scikit-learn tqdm`

Requires python>=3.8

## U-Net

### Training

The code for training the U-Net model can be found in `unet_script.py`. It uses tensorflow to define and train the model with the Nadam optimiser.

The training images should be placed in `./puzzle_corners_1024x768/images-1024x768/`, and masks in `./puzzle_corners_1024x768/masks-1024x768/`

The hyperparameters are as follows:

```
$ shape [(int, int, int)]: The input shape of the images
$ num_classes [int]: The number of classes for segmentation, i.e. background/foreground
$ lr [float]: The learning rate of the model
$ epochs [int]: Number of training epochs. Should be >= size of training dataset
$ savedir [string]: Where to save the final model
```
Dependencies can be installed as follows:

`python3 -m pip install natsort matplotlib glob numpy imagio scikit-image scipy scikit-learn tqdm`

Please refer to https://www.tensorflow.org/install for installation instructions for tensorflow.

Requires python>=3.8


### Testing

The code for obtaining the accuracy results of the trained model can be found in `unet_testing.ipynb`. It performs inference on the testing dataset and calculates the accuracy metrics present in the report. Dependencies are the same as above, and the loaction of the saved model should be defined in `$savedir`