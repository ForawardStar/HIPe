
# The code for "Hierarchical Image Peeling: A Flexible Scale-space Filtering Framework"

## Introduction
The importance of hierarchical image organization has been witnessed by a wide spectrum of applications in computer vision and graphics. Different from image segmentation with the spatial whole-part consideration, this work designs a modern framework for disassembling an image into a family of derived signals from a scale-space perspective. Specifically, we first offer a formal definition of image disassembly. Then, by concerning desired properties, such as peeling hierarchy and structure preservation, we convert the original complex problem into a series of two component separation sub-problems, significantly reducing the complexity. The proposed framework is flexible to both supervised and unsupervised settings. A compact recurrent network, namely hierarchical image peeling net, is customized to efficiently and effectively fulfill the task, which is about 3.5Mb in size, and can handle 1080p images in more than 60 fps per recurrence on a GTX 2080Ti GPU, making it attractive for practical use. Both theoretical findings and experimental results are provided to demonstrate the efficacy of the proposed framework, reveal its superiority over other state-of-the-art alternatives, and show its potential to various applicable scenarios.

## Network Architecture
![Reesuly](img/archf.png)

## Dependnecy
python 3.5, pyTorch >= 1.4.0 (from https://pytorch.org/), numpy, Pillow.
## Usage

### Training
1. Download the dataset you want to use and change the dataset directory. More datatsets can be found https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

2. Starting training using the following command

```python train.py```

If one wants to use the exemplar image which doesn't belong to target domain, just starting training using the following command

```python train_CrossDomain.py```
 
### Testing
1. Put the pre-trained model in your own path and change the checkpoint path of the code
2. Starting testing using the following command

```python test.py```

## Results
![Reesuly](img/exp.png)
![Reesuly](img/ourf.png)
More Results can be found in our website: https://forawardstar.github.io/EDIT-Project-Page/

## Implementation Details
When translating shoes/handbags to edges or translating facades (buildings) to semantic maps, style losses are not needed because edges or semantic maps are exemplar images. Thus, our code in 'train.py' use a if statement to distinguish shoes/handbags to edges and facades to maps from the other domains. Our code sets label = 0 and label = 1 to represent shoes to edges and facades to maps respectively, please change the following code in 'train.py'  if necessary according to your own settings.
```
if label == 3:
    loss_style_AB = criterion_style(fakeB_mean_std, realB_mean_std)
else:
    loss_style_AB = 0
```

## Citation
```
@misc{fu2019edit,
    title={EDIT: Exemplar-Domain Aware Image-to-Image Translation},
    author={Yuanbin Fu and Jiayi Ma and Lin Ma and Xiaojie Guo},
    year={2019},
    eprint={1911.10520},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
