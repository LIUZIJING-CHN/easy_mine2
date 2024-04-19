CUDA_VISIBLE_DEVICES=2 python main_train_attri.py \
--dataset-path "/data/lzj/easy" \
--dataset miniimagenet \
--model resnet12 \
--epochs 100 \
--episodic \
--load-model /data/lzj/easy/result/0818/mini1.pt1 \
--manifold-mixup 0 \
--episodes-per-epoch 2000 \
--gamma 0.3 \
--lr 1e-4 \
--milestones [70,90] \
--skip-epochs 85 \
--batch-size 80 \
--preprocessing ME \
--save-attri-model "/data/lzj/easy/result/1009/attri_model_1.pt" \
--n-shot [1,5] 
# --save-model "result/0905/mini2_finetune.pt" \
# --rotations \
# --cosine \