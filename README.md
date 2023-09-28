# SLiMe: Segment Like Me
**[Paper](https://arxiv.org/abs/2309.03179)**
![SLiMe Method](https://github.com/aliasgharkhani/one_shot_segmentation/blob/public/media/slime_teaser.gif)


PyTorch implementation of SLiMe: Segment Like Me, a 1-shot image segmentation method based on Stable Diffusion. <br><br>
[Aliasghar Khani<sup>1, 2</sup>](https://aliasgharkhani.github.io/), [Saeid Asgari Taghanaki<sup>2</sup>](https://asgsaeid.github.io/), [Aditya Sanghi<sup>2</sup>](https://www.research.autodesk.com/people/aditya-sanghi/), [Ali Mahdavi Amiri<sup>1</sup>](https://www.sfu.ca/~amahdavi/), [Ghassan Hamarneh<sup>1</sup>](https://www.medicalimageanalysis.com/)

<sup><sup>1</sup> Simon Fraser University  <sup>2</sup> Autodesk Research</sup>

# Setup
To begin using SLiMe, you first need to create a virtual environment and install the dependencies using the following commands:
```
python -m venv slime_venv
source slime_venv/bin/activate
pip install -r requirements.txt
```

*** ***For each image and mask pair used for training, validation, or testing with SLiMe, their names should match. Furthermore, the images should be in `PNG` format, while the masks should be in `NumPy` format.*** ***

# SLiMe training
First, create a new folder (e.g., `slime/data/train`) and place the training images along with their corresponding masks in that folder (`slime/data/train`). Then, provide the path to the created training data folder (`slime/data/train`) as an argument to `--train_data_dir`. If you have validation data, which will only be used for checkpoint selection, repeat the same process for the validation data (e.g., place the images and masks in `slime/data/val`) and provide the folder's address as an argument to `--val_data_dir`. However, if you don't have validation data, use the address of the training data folder as an argument for `--val_data_dir`.

Next, place the test images in a separate folder (e.g., `slime/data/test`) and specify the path to this folder using `--test_data_dir`. Additionally, you should define a name for the segmented parts within the training images to be used with the `--parts_to_return` argument, including the background. For instance, if you have segmented the body and head of a dog, you should set `--parts_to_return` to `"background body head"`.

Finally, execute the following command within the slime folder (the main folder obtained after cloning):
```
python -m src.main --dataset sample \
                   --part_names {PARTNAMES} \
                   --train_data_dir {TRAIN_DATA_DIR} \
                   --val_data_dir {TRAIN_DATA_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
                   --train \
```
If you have supplied test images along with their corresponding masks, running this command will display the mean Intersection over Union (mIoU) for each of the segmented parts on the test data. Furthermore, it will save the trained text embeddings in the `slime/outputs/checkpoints` folder and log files in the `slime/outputs/lightning_logs` folder within the `slime` directory.

# Testing with the trained text embeddings
To use the trained text embeddings for testing, run this command:
```
python -m src.main --dataset sample \
                   --checkpoint_dir {CHECKPOINT_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
```
In this command:

- Replace `{CHECKPOINT_DIR}` with the path to the folder where the trained text embeddings are stored. Ensure that only the relevant text embeddings are present in this directory because the code will load all text embeddings from the specified folder.
- Make sure you've placed the test images (and their masks, if available, for calculating mIoU) in a new folder, and provide the path to this folder using the `--test_data_dir` argument.

## Patchifying the Image
To configure the patching of images for validation and testing, you can specify different values for the `--patch_size` and `--num_patches_per_side` parameters. These settings will be used to divide the image into a grid of patches, calculate individual final attention maps (referred to as **WAS-attention** maps), aggregate them, and generate the segmentation mask prediction.

Here's an example of how to include these parameters in your command:
```
python -m src.main --dataset sample \
                   --checkpoint_dir {CHECKPOINT_DIR} \
                   --test_data_dir {TEST_DATA_DIR} \
                   --patch_size {PATCH_SIZE} \
                   --num_patches_per_side {NUM_PATCHES_PER_SIDE}
```
- Replace `{PATCH_SIZE}` with the desired size for each patch.
- Replace `{NUM_PATCHES_PER_SIDE}` with the number of patches you want per side of the image.
By adjusting these values, you can control the patching process for validation and testing, which can be useful for fine-tuning the method's performance on different image sizes or characteristics.

# 1-sample and 10-sample training on datasets
## PASCAL-Part Car
To train and test with the 1-sample setting of SLiMe on the car class of PASCAL-Part, follow these steps:

1. Download the data from this [link](https://drive.google.com/file/d/1PmJQWQleiKlRwTF515A1VhIoQDH4rBtD/view?usp=drive_link) and extract it.
2. Navigate to the `slime` folder.
3. Run the following command, replacing `{path_to_data_folder}` with the path to the folder where you extracted the data (without a backslash at the end):

```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name pascal \
                    --part_names background body light plate wheel window \
                    --train_data_dir $DATADIR/car/train_1 \
                    --val_data_dir $DATADIR/car/train_1 \
                    --test_data_dir $DATADIR/car/test \
                    --min_crop_ratio 0.6 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train
```

For the 10-sample setting, you can modify the command as follows:
```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name pascal \
                    --part_names background body light plate wheel window \
                    --train_data_dir $DATADIR/car/train_10 \
                    --val_data_dir $DATADIR/car/val \
                    --test_data_dir $DATADIR/car/test \
                    --min_crop_ratio 0.6 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train
```

In this case, you should specify `car/train_10` for `--train_data_dir` and `car/val` for `--val_data_dir`.


## PASCAL-Part Horse
To train and test with the 1-sample setting of SLiMe on the horse class of PASCAL-Part, you can follow these steps:

1. Download the data from this [link](https://drive.google.com/file/d/1PmJQWQleiKlRwTF515A1VhIoQDH4rBtD/view?usp=drive_link) and extract it.
2. Navigate to the `slime` folder.
3. Run the following command, replacing `{path_to_data_folder}` with the path to the folder where you extracted the data (without a backslash at the end):

```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name pascal \
                    --part_names background head neck+torso leg tail \
                    --train_data_dir $DATADIR/horse/train_1 \
                    --val_data_dir $DATADIR/horse/train_1 \
                    --test_data_dir $DATADIR/horse/test \
                    --min_crop_ratio 0.8 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train
```

For the 10-sample setting, you can modify the command as follows:

```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name pascal \
                    --part_names background head neck+torso leg tail \
                    --train_data_dir $DATADIR/horse/train_10 \
                    --val_data_dir $DATADIR/horse/val \
                    --test_data_dir $DATADIR/horse/test \
                    --min_crop_ratio 0.8 \
                    --num_patchs_per_side 1 \
                    --patch_size 512 \
                    --train
```

In this case, you should specify `horse/train_10` for `--train_data_dir` and `horse/val` for `--val_data_dir`.

## CelebAMask-HQ
To train and test with the 1-sample setting of SLiMe on CelebAMask-HQ, you can follow these steps:

1. Download the data from this [link](https://drive.google.com/file/d/1PmJQWQleiKlRwTF515A1VhIoQDH4rBtD/view?usp=drive_link) and extract it.
2. Navigate to the `slime` folder.
3. Run the following command, replacing `{path_to_data_folder}` with the path to the folder where you extracted the data (without a backslash at the end):

```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name celeba \
                    --part_names background skin eye mouth nose brow ear neck cloth hair \
                    --train_data_dir $DATADIR/celeba/train_1 \
                    --val_data_dir $DATADIR/celeba/train_1 \
                    --test_data_dir $DATADIR/celeba/test \
                    --min_crop_ratio 0.6 \
                    --train
```

For the 10-sample setting, you can modify the command as follows:

```
DATADIR={path_to_data_folder}
python3 -m src.main --dataset_name celeba \
                    --part_names background skin eye mouth nose brow ear neck cloth hair \
                    --train_data_dir $DATADIR/celeba/train_10 \
                    --val_data_dir $DATADIR/celeba/val \
                    --test_data_dir $DATADIR/celeba/test \
                    --min_crop_ratio 0.6 \
                    --train
```

In this case, you should specify `celeba/train_10` for `--train_data_dir` and `celeba/val` for `--val_data_dir`.

# Trained text embeddings
At this [link](https://drive.google.com/drive/folders/1sA8od8iFbyD2T47A8JsevRf-ExkLV0lT?usp=sharing), we are uploading the text embeddings that we have trained, including the text embeddings we trained for the paper. You can download these text embeddings and use them for testing on your data using the command in [Testing with the trained text embeddings](https://github.com/aliasgharkhani/one_shot_segmentation/tree/master#testing-with-the-trained-text-embeddings) section.
