# Getting Started

## Prepare Data
1. Prepare the COVID-19 dataset:

    We randomly select a subset of patients for `test` and `val` sets.
   ```
   python data_tools/prepare_covid_data.py 
   ```
   Modify the file and rerun to update the train-val-test data split.

2. Prepare the combined dataset:

   ```
   python data_tools/prepare_data.py [--combine_pneumonia]
   ```
   - Class 0: Normal
   - Class 1: Bacterial Pneumonia
   - Class 2: Viral Pneumonia
   - Class 3: COVID-19

## Prepare pretrained-model
`CovidAID` uses the pretrained `CheXNet` model from [here](https://github.com/arnoweng/CheXNet/). We modify the network to classify among 4 classes, while keeping the convolutional layers same. Thus we initialize with `CheXNet` pretrained model weights and fine-tune on top of it.

```
python tools/transfer.py [--combine_pneumonia]
```

## Training
1. Train the classifier layer

    First we train the classifier layer, while freezing the weights of the convolutional layers to be the same as `CheXNet`.
    ```
    python tools/trainer.py --mode train --freeze --checkpoint models/CovidAID_transfered.pth.tar --bs 16 --save <PATH_TO_SAVE_MODELS_FOLDER> [--combine_pneumonia]
    ```

2. Fine tune the convolutional layers

    Next we take the best model from previous step (according to loss), and fine tune the full model. Since we are interested in increasing the recall of `COVID-19`, we specify the `inc_recall` option to `3` (see our paper [paper](http://arxiv.org/abs/2004.09803) for details).
    ```
    python tools/trainer.py --mode train --checkpoint <PATH_TO_BEST_MODEL> --bs 8 --save <PATH_TO_SAVE_MODELS_FOLDER> [--combine_pneumonia]
    ```

## Evaluation
Next we run the best model on the test set to see the results.
```
python tools/trainer.py --mode test --checkpoint <PATH_TO_BEST_MODEL> --cm_path plots/cm_best --roc_path plots/roc_best [--combine_pneumonia]
```

## Inference with trained models
Trained models are available in the `models` directory. 

To run simple inference on a set of images, use:
```
python tools/inference.py --img_dir <IMG_DIR> --checkpoint <BEST_MODEL_PTH> [--combine_pneumonia] [--visualize_dir <OUT_DIR>]
```

## 3-class classification
We also provide functionality of three class classification combining the two types of common pneumonias into a single class. Specify the `--combine_pneumonia` flag to activate this functionality.
