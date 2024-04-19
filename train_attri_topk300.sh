CUDA_VISIBLE_DEVICES=0 python main_train_attri_topk300.py \
--dataset-path "/data/lzj/easy" \
--dataset miniimagenet \
--model resnet12 \
--epochs 100 \
--load-model /data/lzj/easy/result/0818/mini1.pt1 \
--manifold-mixup 0 \
--gamma 0.3 \
--lr 0.01 \
--milestones [50,70] \
--skip-epochs 0 \
--batch-size 80 \
--preprocessing ME \
--save-attri-model "/data/lzj/easy/result/0910/res12frozen_attri_topk300/attri_model_1.pt" \
--n-shot [1,5] \
--episodic \
--episodes-per-epoch 500 
# --save-model "result/0905/mini2_finetune.pt" \
# --rotations \
# --cosine \