# HICVM-Net

VMamba, as a widely used visual backbone, has demonstrated efficient and effective potential in medical image segmentation. However, this model has room for improvement due to its inherent limitations of network structure. Firstly, features in cross-channels are not adequately interacted. Thus, it has suboptimal parameter efficiency. Secondly, features in diagonal region cannot be effectively captured, which hinders model improvement. To address the above mentioned two challenges, a model called hybrid interaction cascaded VMamba (HICVM-Net) is proposed. The core module of the proposed HICVM-Net is pipeline layer, which comprises two collaborative components: cascaded VMamba (CVM) block and hybrid interaction (HI) block. The CVM block establishes a novel channel interaction mechanism that enhances channel interaction efficiency. The HI block efficiently captures diagonal region information, which contributes to model enhancement. Comprehensive experiments conducted on four widely used datasets (ISIC2018, BUSI, Kvasir-SEG and GlaS) demonstrate that HICVM-Net achieves superior model performance across six metrics compared to other popular models, concurrently sustaining low parameters (9.22M) and computational overhead (3.09 GFLOPs).


## Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 

ISIC-2018 (dermoscopy, with 3,694 images)<br>
BUSI (breast ultrasound, with 647 images)<br>
Kvasir-SEG (endoscopy, with 1,000 images)<br> 
GlaS (gland, with 165 images)<br> 


In ISIC 2018 dataset, we adopt the official split configuration, consisting of a training set with 2,594 images, a validation set with 100 images, and a test set with 1,000 images. <br>
In GlaS dataset, we split the dataset into a training set of 85 images and a test set of 80 images. <br>
For other dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.<br>

The dataset path may look like:
```bash
HICVM-Net-main
├── /datasets/
	├── BUSI/
		├── Train_Folder/
		│   ├── img
		│   ├── labelcol
		│
		├── Val_Folder/
		│   ├── img
		│   ├── labelcol
		│
		├── Test_Folder/
			├── img
			├── labelcol
```


## Usage

---

### **Installation**

CUDA Toolkit and CuDNN installation: <br>
CUDA Toolkit:	https://developer.nvidia.com/cuda-toolkit-archive <br>
CuDNN:			https://developer.nvidia.com/rdp/cudnn-archive <br>

Basic environment configuration：
```bash
git clone https://github.com/HvitAska-Eyjafjalla/HICVM-Net
conda create -n env_name python=3.10
conda activate env_name
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

```
Mamba libraries:<br>
causal-conv1d:	https://github.com/Dao-AILab/causal-conv1d <br>
mamba-ssm:		https://github.com/state-spaces/mamba <br>

If you are using the GPU before the NVIDIA RTX 50 series, you can follow this tutorial: (Simplified-Chinese webpage)<br>
https://github.com/AlwaysFHao/Mamba-Install <br>

If you are using the NVIDIA RTX 50 series GPU, you can follow this tutorial: (Simplified-Chinese webpage)<br>
https://blog.csdn.net/yyywxk/article/details/146798627 <br>

### **Training**
```bash
python start.py
```
To run on different setting or different datasets, please modify config_universal.py or config_model.py.


### **Evaluation**
```bash
python test.py
``` 


## Citation

Our repo is useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *IEEE Transactions on Circuits and Systems for Video Technology*.<br>
```bibtex
@ARTICLE{HICVM-Net,
  author  = {Zirui Yan, Shiren Li, Longquan Shao, Yanli Zhang, Serestina Viriri, Qian Dong and Guangguang Yang},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology}
  title   = {HICVM-Net: Hybrid Interaction Cascaded VMamba Network for Medical Image Segmentation},
  year    = {2025}
}
```


## Contact
For technical questions, please contact yanagiama@gmail.com .
