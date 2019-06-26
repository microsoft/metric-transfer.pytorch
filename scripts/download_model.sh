set -x
set -e

base_url="https://frontiers.blob.core.windows.net/metric-transfer"
local_root=checkpoint

mkdir -p log

# you can comment some file if you don't want to download all of them.
echo "downloading pretrained models"
dirname=pretrain_models
mkdir -p ${local_root}/${dirname}
for filename in \
    ckpt_colorization_wrn-28-2.pth.tar \
    ckpt_instance_cifar10_wrn-28-10_89.83.pth.tar \
    ckpt_imagenet32x32_instance_wrn-28-2.pth.tar  \
    ckpt_instance_cifar10_wrn-28-2_82.12.pth.tar \
    ckpt_imagenet32x32_snca_wrn-28-2.pth.tar \
    lemniscate_resnet18.pth.tar \
    ckpt_imagenet32x32_softmax_wrn-28-2.pth.tar \
    lemniscate_resnet50.pth.tar \
    ckpt_instance_cifar10_resnet18_85.69.pth.tar; 
do 
    file=${dirname}/${filename};
    wget ${base_url}/${file} -O ${local_root}/${file} -o log/${dirname}_${filename}.txt --no-clobber &
done
wait 


echo "downloading pre-extracted features"
dirname=train_features_labels_cache
mkdir -p ${local_root}/${dirname}
for filename in \
    colorization_embedding_128.t7 \
    instance_imagenet_val_feature_resnet50.pth.tar \
    instance_imagenet_train_feature_resnet50.pth.tar; 
do 
    file=${dirname}/${filename};
    wget ${base_url}/${file} -O ${local_root}/${file} -o log/${dirname}_${filename}.txt --no-clobber &
done
wait 


echo "downloading pseudo file for cifar10 dataset"
dirname=pseudos
mkdir -p ${local_root}/${dirname}
for filename in \
    colorization_knn_wrn-28-2.tar \
    imagenet32x32_snca_nc_wrn-28-2.tar \
    instance_nc_wrn-28-2.tar \
    colorization_nc_wrn-28-2.tar \
    imagenet32x32_softmax_nc_wrn-28-2.tar \
    imagenet32x32_instance_nc_wrn-28-2.tar \
    instance_knn_wrn-28-2.tar; 
do 
    file=${dirname}/${filename};
    wget ${base_url}/${file} -O ${local_root}/${file} -o log/${dirname}_${filename}.txt --no-clobber &
done
wait 

echo "downloading pseudo file for imagenet dataset"
dirname=pseudos_imagenet/instance_imagenet_nc_resnet50
mkdir -p ${local_root}/${dirname}
for filename in \
    num_labeled_13000.tar \
    num_labeled_26000.tar \
    num_labeled_51000.tar;
do 
    file=${dirname}/${filename};
    wget ${base_url}/${file} -O ${local_root}/${file} -o log/pseudos_imagenet_${filename}.txt --no-clobber &
done
wait 

echo "download finished, extracting"
for folder in pseudos pseudos_imagenet/instance_imagenet_nc_resnet50; do 
(
    cd ${local_root}/${folder};
    for i in $(ls *.tar); do 
        tar xvf $i; 
        rm $i;
    done
)
done

