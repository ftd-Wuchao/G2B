#!/bin/bash

#SBATCH --nodelist=compute003
#SBATCH --mem=20G                 # memory
#SBATCH --gres=gpu:2              # Number of GPU(s): 1 for DTW, 3 for Feature extract.
#SBATCH --time=30-00:00:00          # time (DD-HH:MM:SS) 3 days by default; 5-00:00:00 (5 DAYS) / UNLIMITED;
#SBATCH --ntasks=1                # Number of "tasks”/#nodes (use with distributed parallelism).
#SBATCH --cpus-per-task=6         # Number of CPUs allocated to each task/node (use with shared memory parallelism).

# DTW Templates

# --nodelist=compute001

#hostname
#whoami
echo "//////////////////////////////"

echo 'CUDA_VISIBLE_DEVICES:'
echo $CUDA_VISIBLE_DEVICES
echo "//////////////////////////////"


source activate py181

#
bash train.sh 0,1 \
  --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
  --name dytox_new_order1_mem2000 \
  --data-path /home/share/wuchao/Dataset/IL/ \
  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
  --memory-size 2000 \
  --add_mask

#bash train.sh 0,1 \
#--options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
#--name dytox_new_order1_mem2000 \
#--data-path /home/share/wuchao/Dataset/IL/ \
#--output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#--memory-size 2000 \


#bash train.sh 0,1 \
#  --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
#  --name worelu_dytox_new_order1_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000 \
#  --add_mask


#bash train.sh 0,1 \
#  --options options/data/cifar100_5-5.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
#  --name dytox_new_order1_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000 \
#  --add_mask
#
#bash train.sh 0,1 \
#  --options options/data/cifar100_5-5.yaml options/data/cifar100_order2.yaml options/model/cifar_dytox.yaml \
#  --name dytox_new_order2_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000 \
#  --add_mask

#bash train.sh 0,1 \
#  --options options/data/cifar100_2-2.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
#  --name dytox_new_add_order1_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000 \
#  --add_mask
#
#bash train.sh 0,1 \
#  --options options/data/cifar100_2-2.yaml options/data/cifar100_order2.yaml options/model/cifar_dytox.yaml \
#  --name dytox_new_order2_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000 \
#  --add_mask


#bash train.sh 0,1\
#  --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
#  --name dytox_new_gpu2_convs_order1_mem2000 \
#  --data-path /home/share/wuchao/Dataset/IL/ \
#  --output-basedir /home/share/wuchao/IL/Dytox/checkpoints \
#  --memory-size 2000
#  --rehearsal-modes 2


#bash train.sh 0,1 \
#    --options options/data/imagenet100_10-10.yaml options/data/imagenet100_order1.yaml options/model/imagenet_dytox.yaml \
#    --name dytox_new_gpu2_order1_mem2000 \
#    --data-path /home/share/wuchao/Dataset/IL/Imagenet/ \
#    --resume /home/share/wuchao/IL/Dytox/checkpoints \
#    --memory-size 2000 \
#    --save-every-epoch 30 \
#    --add_mask


#bash train.sh 0,1 \
#    --options options/data/imagenet1000_100-100.yaml options/data/imagenet1000_order1.yaml options/model/imagenet_dytox.yaml \
#    --name dytox_new \
#    --data-path /home/share/wuchao/Dataset/IL/Imagenet/ \
#    --resume /home/share/wuchao/IL/Dytox/checkpoints \
#    --memory-size 20000 \
#    --add_mask