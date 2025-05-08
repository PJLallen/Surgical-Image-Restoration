# Surgical-Image-Restoration
SurgClean Benchmark for Surgical Image Restoration.

### Requirements

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA 11.6](https://developer.nvidia.com/cuda-downloads)
- Linux (We have not tested on Windows)

### Installation

1. Clone the repo

    ```bash
    git clone https://github.com/PJLallen/Surgical-Image-Restoration.git
    ```

1. Install dependent packages

    ```bash
    cd Surgical-Image-Restoration
    pip install -r requirements.txt
    python setup.py develop
    ```
## Data Preparing
### Data Download

|     | Kaggle | Number | Description|
| :--- | :--: | :---- | ---- |
| SurgClean | [link](https://kaggle.com/datasets/5bad41858571a3a9ea2f65a50c1d1d81c71956cc966c5b6ab96a42fa46418d78) | 1,020 | SurgClean involves multi-type image restoration tasks, i.e., desmoking, defogging, and desplashing. It comprises 1,020 multi-type endoscopic images with varying degradation types and corresponding adjacent clean frames as unaligned paired labels.|

### SurgClean structure

```
├── SurgClean Dataset
	├── Defog
		├── train
			├── gt
			├── input
		├── test
			├── gt
			├── input
	├── Desmoke
	```
	├── Desplash
	```
├── SurgClean Dataset_Fine-grain Division
	├── Defog_Level
		├── test
			├── gt
				├── Level-1
				├── Level-2
				├── Level-3
				├── Level-4
			├── input
				├── Level-1
				├── Level-2
				├── Level-3
				├── Level-4
	├── Desmoke_Level
	```
	├── Desplash_Category
			├── gt
				├── bile
				├── blood
				├── fat
				├── tissue fluid
			├── input
				├── bile
				├── blood
				├── fat
				├── tissue fluid
```


## Pretrained Model

The inference code based on Uformer is released Now. Your can download the pretrained checkpoints from the following links. Please place it under the `experiments` folder and unzip it, then you can run the `test.py` for inference. We provide two checkpoints for models training on Flare7K, the model in the folder `uformer` can help remove both the reflective flares and scattering flares. The `uformer_noreflection` one can only help remove the scattering flares but is more robust. Now, we prefer the users to test our new model trained on Flare7K++, it can achieve better results and more realistic light source.

| Training Data       |                        Baidu Netdisk                         |                         Google Drive                         |
| :------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Flare7K             | [link](https://pan.baidu.com/s/1EJSYIbbQe5SZYiNIcvrmNQ?pwd=xui4) | [link](https://drive.google.com/file/d/1uFzIBNxfq-82GTBQZ_5EE9jgDh79HVLy/view?usp=sharing) |
| Flare7K++ (**new**) | [link](https://pan.baidu.com/s/1lC4zSda5O2aUtMPlZ9sRiw?pwd=nips)  | [link](https://drive.google.com/file/d/17AX9BJ-GS0in9Ey7vw3BVPISm67Rpzho/view?usp=sharing)|

## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

### Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_train.sh 8 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml


## Testing Commands

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

### Distributed Testing

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_test.sh 8 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml  --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

