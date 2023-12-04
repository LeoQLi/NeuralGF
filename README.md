# NeuralGF: Unsupervised Point Normal Estimation by Learning Neural Gradient Function (NeurIPS 2023)

### **[Project](https://leoqli.github.io/NeuralGF/) | [arXiv](https://arxiv.org/abs/2311.00389)**

Normal estimation for 3D point clouds is a fundamental task in 3D geometry processing. The state-of-the-art methods rely on priors of fitting local surfaces learned from normal supervision. However, normal supervision in benchmarks comes from synthetic shapes and is usually not available from real scans, thereby limiting the learned priors of these methods. In addition, normal orientation consistency across shapes remains difficult to achieve without a separate post-processing procedure. To resolve these issues, we propose a novel method for estimating oriented normals directly from point clouds without using ground truth normals as supervision. We achieve this by introducing a new paradigm for learning neural gradient functions, which encourages the neural network to fit the input point clouds and yield unit-norm gradients at the points. Specifically, we introduce loss functions to facilitate query points to iteratively reach the moving targets and aggregate onto the approximated surface, thereby learning a global surface representation of the data. Meanwhile, we incorporate gradients into the surface approximation to measure the minimum signed deviation of queries, resulting in a consistent gradient field associated with the surface. These techniques lead to our deep unsupervised oriented normal estimator that is robust to noise, outliers and density variations. Our excellent results on widely used benchmarks demonstrate that our method can learn more accurate normals for both unoriented and oriented normal estimation tasks than the latest methods.

## Requirements
The code is implemented in the following environment settings:
- Ubuntu 20.04
- CUDA 11.7
- Python 3.8
- Pytorch 1.9
- Pytorch3d 0.6
- Numpy 1.19
- Scipy 1.6

We train and test our code on an NVIDIA 3090 Ti GPU.

## Dataset
The datasets used in our paper can be downloaded from [Here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
Unzip them to a folder `***/dataset/` and set the path value of `dataset_root` in `train_test.py`.
The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──FamousShape/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
```
python train_test.py --mode=train --gpu=0 --data_set=***
```
You need to set `data_set` according to the used dataset. The trained models will be save in `./log/***/`.

## Test
Our pre-trained models can be downloaded from [Here](https://drive.google.com/drive/folders/1ZTlNSpou1KU7KCRyYAo1BwNcu-2NytO5?usp=sharing).

To test on the PCPNet dataset using the provided models, simply run:
```
python train_test.py --mode=test --gpu=0 --data_set=PCPNet --ckpt_dir=231007_140818_PCPNet --ckpt_iter=20000
```
The predicted normals and evaluation results will be saved in `./log/231007_140818_PCPNet/test_20000/`.

To save the predicted normals, you need to set `save_normal_npy` or `save_normal_xyz` to True.
To save the reconstructed surfaces, you need to set `save_mesh` to True.

## Citation
If you find our work useful in your research, please cite our paper:

    @inproceedings{li2023neuralgf,
      title={{NeuralGF}: Unsupervised Point Normal Estimation by Learning Neural Gradient Function},
      author={Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
      year={2023}
    }

