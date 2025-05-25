# [2025] Surgical Image Restoration Benchmark
Official Implementation of "[Benchmarking Laparoscopic Surgical Image Restoration and Beyond]()"

[Jialun Pei](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en), [Diandian Guo](https://scholar.google.com/citations?user=yXycwhIAAAAJ&hl=zh-CN&oi=ao), [Donghui Yang](), [Zhixi Li](), [Yuxin Feng](), [Long Ma](https://scholar.google.com/citations?user=QeCRo9sAAAAJ&hl=zh-CN&oi=ao), [Bo Du](https://scholar.google.com/citations?user=Shy1gnMAAAAJ&hl=zh-CN&oi=ao), and [Pheng-Ann Heng](https://scholar.google.com/citations?user=OFdytjoAAAAJ&hl=zh-CN)

**Contact:** peijialun@gmail.com, malone94319@gmail.com

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

|     | Kaggle | Description|
| :--- | :--: | ---- |
| SurgClean | [link]() | SurgClean involves multi-type image restoration tasks, i.e., desmoking, defogging, and desplashing. It comprises 1,020 multi-type endoscopic images with varying degradation types and corresponding adjacent clean frames as unaligned paired labels.|

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


## Pretrained Models

| Training Data       |                   One Drive                     |
| :------------------ | :----------------------------------------------------------: |
| Restormer    | [link]() |
| FocalNet | [link]()  |
| ConvIR    | [link]() |
| Fourmer | [link]()  |
| MambaIR    | [link]() |
| Histoformer | [link]()  |
| RAMiT    | [link]() |
| AMIR | [link]()  |
| AST    | [link]() |
| X-Restormer | [link]()  |
| SFHformer   | [link]() |
|  MambaIRv2 | [link]()  |

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

## 📚 Citation

If this helps you, please cite this work:

```bibtex
@inproceedings{pei2024restoration,
  title={Benchmarking Laparoscopic Surgical Image Restoration and Beyond},
  author={Pei, Jialun and Guo, Diandian and Yang, Donghui and Li, Zhixi and Feng, Yuxin and Ma, Long and Du, Bo and Heng, Pheng-Ann},
  booktitle={arXiv},
  year={2025}
}
```


