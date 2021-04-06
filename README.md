
# The code for "Hierarchical Image Peeling: A Flexible Scale-space Filtering Framework"

## Introduction
The importance of hierarchical image organization has been witnessed by a wide spectrum of applications in computer vision and graphics. Different from image segmentation with the spatial whole-part consideration, this work designs a modern framework for disassembling an image into a family of derived signals from a scale-space perspective. Specifically, we first offer a formal definition of image disassembly. Then, by concerning desired properties, such as peeling hierarchy and structure preservation, we convert the original complex problem into a series of two component separation sub-problems, significantly reducing the complexity. The proposed framework is flexible to both supervised and unsupervised settings. A compact recurrent network, namely hierarchical image peeling net, is customized to efficiently and effectively fulfill the task, which is about 3.5Mb in size, and can handle 1080p images in more than 60 fps per recurrence on a GTX 2080Ti GPU, making it attractive for practical use. Both theoretical findings and experimental results are provided to demonstrate the efficacy of the proposed framework, reveal its superiority over other state-of-the-art alternatives, and show its potential to various applicable scenarios.

## Network Architecture
![Reesuly](img/arch.png)

## Dependnecy
python 3.5, pyTorch >= 1.4.0 (from https://pytorch.org/), numpy, Pillow.
## Usage

### Training
Before training and testing, the data path and output path should be properly changed.
1. Train the HIPe-Guider by running "main.py" existed in the FileFolder named "HIPe-Guider"

2. Train the HIPe-Peeler by running "main.py" existed in the FileFolder named "HIPe-Peeler"

### Testing
We provide test images in FileFolder named "example", and your can change the file path to your own dataset.
1. Run "test.py" existed in the FileFolder named "HIPe-Guider" to generate multi-scale edges as guidance

2. Run "test_smooth.py" existed in the FileFolder named "HIPe-Peeler" to peel the input image guided by the generted edges


## Results
Input image             |  Multi-scale edge guidance
:-------------------------:|:-------------------------:
![](img/building3.png)  |  ![](img/building3_edge.png)

Filtered result             |  1-D signals of intensity corresponding to the rows indicated by the yellow arrow in input
:-------------------------:|:-------------------------:
![](img/building3_smooth.png)  |  ![](img/Plot_firstpic2.png)


## Citation
```
@misc{yuanbin2021hierarchical,
      title={Hierarchical Image Peeling: A Flexible Scale-space Filtering Framework}, 
      author={Fu Yuanbin and Guoxiaojie and Hu Qiming and Lin Di and Ma Jiayi and Ling Haibin},
      year={2021},
      eprint={2104.01534},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
