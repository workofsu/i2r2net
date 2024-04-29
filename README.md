## [Iterative image rain removal network using consecutive residual long short-term memory](https://doi.org/10.1016/j.neucom.2024.127752)

### Abstract
Image rain removal is designed to effectively separate rain streaks from the background image layer. However, 
rain streaks in real-world scenarios vary in density, shape, and direction, making it difficult to decompose rainy 
images into clean backgrounds and rain layers. In this study, we introduce an iterative framework for image 
deraining to progressively enhance rainy images using a residual long short-term memory structure. The overall 
network comprises a multidomain residue channel, a fusion module, and a consecutive residual long short-term 
memory. We introduce multidomain residue channels by computing them in both the image and wavelet low- 
frequency domains. We propose a fusion module to combine the residue channel for guidance with wavelet 
domain features for rain removal. We also propose a feature extraction module based on successive residual long 
short-term memory to extract the main features in the wavelet domain. An iterative image restoration framework 
comprising three primary modules is introduced to progressively enhance rainy images. To evaluate the performance of the proposed approach, we conduct experiments using widely used benchmarks. The results 
demonstrate that our method outperforms state-of-the-art methods in image rain removal

### Requirement
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5 For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~

### Download the dataset and model

### Training

After setting the data_dir and valid_data in main.py, then run

~~~
python main.py
~~~

### Testing
After setting the data_dir and test_model in test.py, then run
~~~
python test.py
~~~

the resulting images will be saved.

Next, run the matlab file to get the scores.

### Citation
if you find this project useful for your research, please consider citing:
~~~
@article{park2024iterative,
  title={Iterative Image Rain Removal Network Using Consecutive Residual Long Short-Term Memory},
  author={Park, Su Yeon and Park, Tae Hee and Eom, Il Kyu},
  journal={Neurocomputing},
  pages={127752},
  year={2024},
  publisher={Elsevier}
}
~~~




