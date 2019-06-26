## Deep Metric Transfer for Label Propagation with Limited Annotated Data

This repo contains the pytorch implementation for the semi-supervised learning paper [(arxiv)](https://arxiv.org/abs/1812.08781).

```latex
@inproceedings{liu2018deep,
  title={Deep Metric Transfer for Label Propagation with Limited Annotated Data},
  author={Liu, Bin and Wu, Zhirong and Hu, Han and Lin, Stephen},
  journal={arXiv preprint arXiv:1812.08781},
  year={2018}
}
```

## Requirements

* Python3: Anaconda is recommended because it already contains a lot of packages:
* `pytorch>=1.0`: Refer to https://pytorch.org/get-started/locally/
* other packages: `pip install tensorboardX tensorboard easydict scikit-image`

## Highlight

- We formulate semi-supervised learning from a completely different metric transfer perspective.
- Enjoys the benefit of recent advances self-supervised learning.
- We hope to draw more attention to unsupervised pretraining for other tasks.

## Quick start

* Clone this repo: `git clone git@github.com:bl0/metric-transfer.pytorch.git && cd metric-transfer.pytorch`

* Install pytorch and other packages listed in requirements

* Download pretrained models and precomputed pseudo labels: `bash scripts/download_model.sh` . Make sure the `checkpoint` folder looks like this:

  ```
  checkpoint
  |-- pretrain_models
  |   |-- ckpt_instance_cifar10_wrn-28-2_82.12.pth.tar
  |   |-- ... other files
  |   `-- lemniscate_resnet50.pth.tar
  |-- pseudos
  |   |-- instance_nc_wrn-28-2
  |   |   |-- 50.pth.tar
  |   |   |-- ... other files
  |   |   `-- 8000.pth.tar
  |   `-- ... other folders 
  `-- pseudos_imagenet
      `-- instance_imagenet_nc_resnet50
          |-- num_labeled_13000
          |   |-- 10_0.pth.tar
          |   |-- ... other files
          |   `-- 10_9.pth.tar
          `-- ... other folders 
  ```

* Supervised finetune on cifar10 dataset or Imagenet dataset. The cifar dataset will be downloaded automatically. For imagenet, refer to [here](https://github.com/pytorch/examples/tree/master/imagenet) for details of data preparation.

  ```bash
  # Finetune on cifar
  python cifar-semi.py \
  	--gpus 0 \
  	--num-labeled 250 \
  	--pseudo-file checkpoint/pseudos/instance_nc_wrn-28-2/250.pth.tar \
  	--resume checkpoint/pretrain_models/ckpt_instance_cifar10_wrn-28-2_82.12.pth.tar \
   	--pseudo-ratio 0.2
   	
  # For imagenet
  n_labeled=13000  # 1% labeled data
  pseudo_ratio=0.1  # use top 10% pseudo label
  data_dir=/path/to/imagenet/dir
  
  python imagenet-semi.py \
      --arch resnet50 \
      --gpus 0,1,2,3 \
      --num-labeled ${n_labeled} \
      --data-dir ${data_dir} \
      --pretrained checkpoint/pretrain_models/lemniscate_resnet50.pth.tar  \
      --pseudo-dir checkpoint/pseudos_imagenet/instance_imagenet_nc_resnet50/num_labeled_${n_labeled} \
      --pseudo-ratio ${pseudo_ratio} \
  ```

## Usage

The proposed method contains three main steps: metric pretraining, label propagation, and supervised finetune.

### Metric pretraining

The metric pretraining can be unsupervised or supervised, from the same or different dataset. 

We provide code for [instance discrimination](https://arxiv.org/abs/1805.01978), which is borrowed from the [original pytorch release](https://github.com/zhirongw/lemniscate.pytorch) of instance discrimination. You can run the following command in root director of code to train the instance discrimination on cifar10 dataset:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0 python unsupervised/cifar.py \
	--lr-scheduler cosine-with-restart \
	--epochs 1270
```

For other metric or imagenet dataset, such as colorization on cifar10 dataset, or instance discrimination on imagenet datset, ref to offical released code: [colorization](https://github.com/richzhang/colorization), [instance discrimination](https://github.com/zhirongw/lemniscate.pytorch). We also provide the pretrained weight. Refer to `scripts/download_model.sh` for more details.

### Label propagation

We then can propagation the label using the trained metric from the few labeled examples to a vast collection of unannotated images.

We consider two propagation algorithms: K-nearest neighbors(i.e. **knn**) and spectral clustering(also called normalized cut, i.e **nc**). The implementation is in `notebooks` folder, which is in jupyter notebook format. You can simplely run the notebook to load the weight of metric pretraining approach and propagate to get the pseudo label.

We alse provide the pseudo label for cifar10 and imagenet dataset. Refer to `scripts/download_model.sh` for more details.

### Supervised finetune

With the estimated pseudo labels on the unlabeled data, we can train a classifier with more data. For simplicity, we omit the confidence weighted supervised training in the current version. Instead, we only use a portion of the most confident pseudo label to training.

Refer to quickstart part for more command instruction.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Contact

For any questions, please feel free to create a new issue or reach 
```
Bin Liu: liubinthss@gmail.com
```