CUDA_VISIBLE_DEVICES=3 python main_res50.py \
--dataset-path "/data1/lzj/easy_mine" \
--dataset cifarfs \
--model resnet12 \
--epochs 100 \
--episodic \
--load-model /data1/lzj/easy_mine/pretrained_weight/r-50-1000ep.pth.tar \
--manifold-mixup 0 \
--episodes-per-epoch 1000 \
--gamma 0.3 \
--lr 3e-3 \
--milestones 100 \
--skip-epochs 0 \
--batch-size 80 \
--save-model "20240103_cifarfs_3e-3_1shot_training.pt" \
--n-shot [1,5] \
--preprocessing E \
--feat-topk \
--topk 500 \
--cosine