CUDA_VISIBLE_DEVICES=0 python main_train_attri_beginning.py \
--dataset-path "/data/lzj/easy" \
--dataset miniimagenet \
--model resnet12 \
--epochs 100 \
--manifold-mixup 0 \
--gamma 0.5 \
--milestones [70,90] \
--skip-epochs 90 \
--batch-size 80 \
--load-model /data/lzj/easy/result/0914/pretrain100/mini1_pretrain_100.pt \
--preprocessing ME \
--save-model "/data/lzj/easy/result/0914/finetune/mini_finetune_1.pt" \
--n-shot [1,5] \
--episodes-per-epoch 2000 \
--episodic \
--lr 1e-5 \
# --rotations \
# --cosine \
# --save-attri-model "/data/lzj/easy/result/0914/finetune/attri_model_1.pt" \

