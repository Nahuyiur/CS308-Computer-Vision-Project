#!/bin/bash
#SBATCH -o /lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/logs/VLM_test.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=a100     # 作业提交的指定分区队列为titan
#SBATCH --qos=a100            # 指定作业的QOS
#SBATCH -J VLM_test       
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

source ~/.bashrc
conda activate nlp  

export PYTHONPATH=$PYTHONPATH:/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning

nvidia-smi

python scripts/test.py
