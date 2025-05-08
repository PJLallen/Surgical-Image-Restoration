# Surgical-Image-Restoration
SurgClean Benchmark for Surgical Image Restoration.


### Installation

1. Clone the repo

    ```bash
    git clone https://github.com/PJLallen/Surgical-Image-Restoration.git
    ```

1. Install dependent packages

    ```bash
    cd Flare7K
    pip install -r requirements.txt
    ```

### Data Download

|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| SurgClean | [link](https://kaggle.com/datasets/5bad41858571a3a9ea2f65a50c1d1d81c71956cc966c5b6ab96a42fa46418d78) | [link](https://drive.google.com/file/d/1PPXWxn7gYvqwHX301SuWmjI7IUUtqxab/view) | 7,962 | Flare7K++ consists of Flare7K and Flare-R. Flare7K offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares. Flare-R offers 962 real-captured flare patterns. |

### Paired Data Generation

We provide a on-the-fly dataloader function and a flare-corrupted/flare-free pairs generation script in this repository. To use this function, please put the Flare7K dataset and 24K Flickr dataset on the same path with the `generate_flare.ipynb` file.

If you only want to generate the flare-corrupted image without reflective flare, you can comment out the following line:
```
# flare_image_loader.load_reflective_flare('Flare7K','Flare7k/Reflective_Flare')
```


### Pretrained Model

The inference code based on Uformer is released Now. Your can download the pretrained checkpoints from the following links. Please place it under the `experiments` folder and unzip it, then you can run the `test.py` for inference. We provide two checkpoints for models training on Flare7K, the model in the folder `uformer` can help remove both the reflective flares and scattering flares. The `uformer_noreflection` one can only help remove the scattering flares but is more robust. Now, we prefer the users to test our new model trained on Flare7K++, it can achieve better results and more realistic light source.

| Training Data       |                        Baidu Netdisk                         |                         Google Drive                         |
| :------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Flare7K             | [link](https://pan.baidu.com/s/1EJSYIbbQe5SZYiNIcvrmNQ?pwd=xui4) | [link](https://drive.google.com/file/d/1uFzIBNxfq-82GTBQZ_5EE9jgDh79HVLy/view?usp=sharing) |
| Flare7K++ (**new**) | [link](https://pan.baidu.com/s/1lC4zSda5O2aUtMPlZ9sRiw?pwd=nips)  | [link](https://drive.google.com/file/d/17AX9BJ-GS0in9Ey7vw3BVPISm67Rpzho/view?usp=sharing)|

### Inference Code
To estimate the flare-free images with our checkpoint pretrained on Flare7K++, you can run the `test.py` or `test_large.py` (for image larger than 512*512) by using:
```
python test_large.py --input dataset/Flare7Kpp/test_data/real/input --output result/test_real/flare7kpp/ --model_path experiments/flare7kpp/net_g_last.pth --flare7kpp
```
If you use our checkpoint pretrained on Flare7K, please run:
```
python test_large.py --input dataset/Flare7Kpp/test_data/real/input --output result/test_real/flare7k/ --model_path experiments/flare7k/net_g_last.pth
```

### Evaluation Code
To calculate different metrics with our pretrained model, you can run the `evaluate.py` by using:
```
python evaluate.py --input result/blend/ --gt dataset/Flare7Kpp/test_data/real/gt/ --mask dataset/Flare7Kpp/test_data/real/mask/
```

### Training model

**Training with single GPU**

To train a model with your own data/model, you can edit the `options/uformer_flare7k_option.yml` and run the following codes. You can also add `--debug` command to start the debug mode:

```
python basicsr/train.py -opt options/uformer_flare7k_option.yml
```
If you want to use Flare7K++ for training, please use:
```
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

**Training with multiple GPU**

You can run the following command for the multiple GPU tranining:

```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/uformer_flare7k_option.yml
```
If you want to use Flare7K++ for training, please use:
```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/uformer_flare7kpp_baseline_option.yml
```

### Flare7k++ structure

```
├── Flare7K
    ├── Reflective_Flare 
    ├── Scattering_Flare
         ├── Compound_Flare
         ├── Glare_with_shimmer
         ├── Core
         ├── Light_Source
         ├── Streak
├── Flare-R
	├── Compound_Flare
	├── Light_Source
├── test_data
     ├── real
          ├── input
          ├── gt
          ├── mask
     ├── synthetic
          ├── input
          ├── gt
	  ├── mask

```
