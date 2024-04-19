### global imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import sys
print("Using pytorch version: " + torch.__version__)

### local imports
print("Importing local files: ", end = '')
from args import args
from utils import *
import datasets
import my_few_shot_eval_word_embed, my_few_shot_eval_word_embed_supp
import resnet
import wideresnet
import resnet12
import s2m2
import mlp
from torchmetrics.functional import pairwise_cosine_similarity
print("models.")
torch.autograd.set_detect_anomaly(True)
if args.ema > 0:
    from torch_ema import ExponentialMovingAverage

if args.wandb:
    import wandb

# attri_bank = torch.load('/data/lzj/easy_mine/word_embedding/FC100/train_vocab_vec.pkl')
# class_relations = torch.load('/data/lzj/easy_mine/word_embedding/FC100/train_vec_simi.pkl')
attri_bank = torch.load('/data1/lzj/easy_mine/word_embedding/miniimagenet/train_vocab_vec.pkl')
class_relations = torch.load('/data1/lzj/easy_mine/word_embedding/miniimagenet/train_vec_simi.pkl')
# attri_bank = torch.load('/data/lzj/easy_mine/word_embedding/cifarfs/train_vocab_vec.pkl')
# class_relations = torch.load('/data/lzj/easy_mine/word_embedding/cifarfs/train_vec_simi.pkl')
### global variables that are used by the train function
if args.attri_bank != '':
    attri_bank = torch.load(args.attri_bank)
if args.class_relations != '':
    class_relations = torch.load(args.class_relations)
last_update, criterion = 0, torch.nn.CrossEntropyLoss()

# criterion_back = torch.nn.MSELoss()
criterion_attri = torch.nn.MSELoss()
criterion_relation = torch.nn.MSELoss()
# def criterion_attri(output_feature, att_feature):
#     output_feature = output_feature / output_feature.norm(dim=1).view(output_feature.shape[0], -1)
#     # print(output_feature.shape)
#     att_feature = att_feature / att_feature.norm(dim=1).view(att_feature.shape[0], -1)
#     cos_simi = (torch.diag(torch.matmul(output_feature, att_feature.transpose(0, 1)))+1)/2
#     # print(cos_simi.shape)
#     loss_all = 1 - cos_simi
#     loss = loss_all.mean(dim=0)
#     # print(loss.shape)
#     return loss

### function to either use criterion based on output and target or criterion_episodic based on features and target
def crit(output, features, target):
    if args.episodic:
        return criterion_episodic(features, target)
    else:
        if args.label_smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes = num_classes, smoothing = args.label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        return criterion(output, target)

import torch


def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    # h_emb (N, M)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix

### main train function
def train(model, train_loader, optimizer, epoch, scheduler, mixup = False, mm = False):
    model.train()
    global last_update
    losses, total = 0., 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
            
        data, target = data.to(args.device), target.to(args.device)
        attri_target = attri_bank[target].to(args.device)
        # print(attri_target.shape)
        # reset gradients
        
        ## 读取标签，读取构成相应标签矩阵
        class_batch = torch.unique(target)
        simi_target = torch.zeros(len(class_batch), len(class_batch)).to(args.device)
        for i in range(len(class_batch)):
            for j in range(len(class_batch)):
                simi_target[i,j] = class_relations[i,j]
        
        # diag = torch.diag(simi_target)  
        # diag = torch.diag_embed(diag)
        # simi_target = (simi_target - diag).to(args.device)
        ##
        optimizer.zero_grad()

        if mm: # as in method S2M2R, to be used in combination with rotations
            # if you do not understand what I just wrote, then just ignore this option, it might be better for now
            new_chunks = []
            sizes = torch.chunk(target, len(args.devices))
            for i in range(len(args.devices)):
                new_chunks.append(torch.randperm(sizes[i].shape[0]))
            index_mixup = torch.cat(new_chunks, dim = 0)
            lam = np.random.beta(2, 2)
            output, output_attri, features = model(data, index_mixup = index_mixup, lam = lam)
            # output, output_attri, back_feat, features = model(data, index_mixup = index_mixup, lam = lam)

            if args.rotations:
                output, _ = output
            loss_mm = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
            # 定义匹配损失
            loss_mm += lam * criterion_attri(output_attri, attri_target)+ (1-lam) * criterion_attri(output_attri, attri_target[index_mixup])
            # 定义传回特征损失
            # loss_mm += 0.5 * criterion_back(back_feat, features)
            
            loss_mm.backward()

        if args.rotations: # generate self-supervised rotations for improved universality of feature vectors
            bs = data.shape[0] // 4
            target_rot = torch.LongTensor(data.shape[0]).to(args.device)
            target_rot[:bs] = 0
            data[bs:] = data[bs:].transpose(3,2).flip(2)
            target_rot[bs:2*bs] = 1
            data[2*bs:] = data[2*bs:].transpose(3,2).flip(2)
            target_rot[2*bs:3*bs] = 2
            data[3*bs:] = data[3*bs:].transpose(3,2).flip(2)
            target_rot[3*bs:] = 3

        if mixup and args.mm: # mixup or manifold_mixup
            index_mixup = torch.randperm(data.shape[0])
            lam = random.random()            
            if args.mm:
                output, features = model(data, index_mixup = index_mixup, lam = lam)
            else:
                data_mixed = lam * data + (1 - lam) * data[index_mixup]
                output, features = model(data_mixed)
            if args.rotations:
                output, output_rot = output
                loss = ((lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])) + (lam * crit(output_rot, features, target_rot) + (1 - lam) * crit(output_rot, features, target_rot[index_mixup]))) / 2
            else:
                loss = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
        else:
            output, output_attri, features = model(data)
            # output, output_attri, back_feat, features = model(data)
            ### 对同类特征进行合并
            loss_relation = 0
            # for i in range(len(class_batch)):
            #     class_proto_1 = features[target==class_batch[i]].mean(dim=0)
            #     # class_proto_1 = class_proto_1 / class_proto_1.norm(dim=-1)
            #     for j in range(i+1, len(class_batch)):
            #         class_proto_2 = features[target==class_batch[j]].mean(dim=0)
            #         # class_proto_2 = class_proto_2 / class_proto_2.norm(dim=-1)
            #         # class_relation = torch.dot(class_proto_1, class_proto_2)
            #         class_relation = torch.cosine_similarity(class_proto_1, class_proto_2, dim=-1)
            #         loss_relation += criterion_relation(class_relation, simi_target[i,j])
                    
            #         # print(cos_simi.shape)
            # sum_total = ((1+len(class_batch))*len(class_batch))/2
            # loss_relation /= sum_total
            # protos_diag = torch.diag(protos_simi_batch)
            # protos_diag = torch.diag_embed(protos_diag)
            # protos_simi_batch = protos_simi_batch - protos_diag
            
            ##
            # sample_relation = calculate_cosine_similarity_matrix(features)
            # # target_simi = torch.zeros_like(sample_relation).to(args.device)
            # # loss_relation = criterion_relation(sample_relation, target_simi)
            # for i in range(len(class_batch)):
            #     class_1_target = class_batch[i]
            #     class_1_relation = sample_relation[target==class_1_target]
            #     for j in range(i+1, len(class_batch)):
            #         class_2_target = class_batch[j]
            #         class_12_relation_matrix = class_1_relation[:, target==class_2_target]
            #         class_12_relation = class_12_relation_matrix.mean(dim=-1).mean(dim=-1)
            #         loss_relation += criterion_relation(class_12_relation, simi_target[i,j])
            
            # sum_total = ((1+len(class_batch))*len(class_batch))/2
            # loss_relation /= sum_total
            ##
            
            ## 用feature进行对比
            class_protos_list = []
            for i in range(len(class_batch)):
                class_proto = features[target==class_batch[i]].mean(dim=0, keepdim=True)
                class_protos_list.append(class_proto)
            class_protos = torch.cat(class_protos_list, dim=0)
            proto_relations = calculate_cosine_similarity_matrix(class_protos)
            loss_relation = criterion_relation(proto_relations, simi_target)
            ##
            ## 用映射特征进行对比
            # class_protos_list = []
            # for i in range(len(class_batch)):
            #     class_proto = output_attri[target==class_batch[i]].mean(dim=0, keepdim=True)
            #     class_protos_list.append(class_proto)
            # class_protos = torch.cat(class_protos_list, dim=0)
            # proto_relations = calculate_cosine_similarity_matrix(class_protos)
            # loss_relation = criterion_relation(proto_relations, simi_target)
            ##
            ###
            if args.rotations:
                output, output_rot = output
                loss = 0.5 * crit(output, features, target) + 0.5 * crit(output_rot, features, target_rot) + args.attriloss_weight * criterion_attri(output_attri, attri_target)
                # loss = 0.5 * crit(output, features, target) + 0.5 * crit(output_rot, features, target_rot) + 0.5 * criterion_attri(output_attri, attri_target)
                # loss = 0.5 * crit(output, features, target) + 0.5 * crit(output_rot, features, target_rot) + 0.5 * criterion_attri(output_attri, attri_target) + 0.5 * criterion_back(back_feat, features)         
                loss += args.relationloss_weight * loss_relation      
                # loss += 0.5 * loss_relation      
            else:
                loss = 0.5 * crit(output, features, target) + args.attriloss_weight * criterion_attri(output_attri, attri_target)
                # loss = 0.5 * crit(output, features, target) + 0.5 * criterion_attri(output_attri, attri_target)
                # loss = 0.5 * crit(output, features, target) + 0.5 * criterion_attri(output_attri, attri_target) + 0.5 * criterion_back(back_feat, features)
                loss += args.relationloss_weight * loss_relation
                # loss += 0.5 * loss_relation

        # backprop loss
        loss.backward()
            
        losses += loss.item() * data.shape[0]
        total += data.shape[0]
        # update parameters
        optimizer.step()
        scheduler.step()
        if args.ema > 0:
            ema.update()

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)
        # print advances if at least 100ms have passed since last print
        if (batch_idx + 1 == length) or (time.time() - last_update > 0.1) and not args.quiet:
            if batch_idx + 1 < length:
                print("\r{:4d} {:4d} / {:4d} loss: {:.5f} time: {:s} lr: {:.5f} ".format(epoch, 1 + batch_idx, length, losses / total, format_time(time.time() - start_time), float(scheduler.get_last_lr()[0])), end = "")
            else:
                print("\r{:4d} loss: {:.5f} ".format(epoch, losses / total), end = '')
            last_update = time.time()

        if few_shot and total >= args.dataset_size and args.dataset_size > 0:
            break
            
    if args.wandb:
        wandb.log({"epoch":epoch, "train_loss": losses / total})

    # return train_loss
    return { "train_loss" : losses / total}

# function to compute accuracy in the case of standard classification
def test(model, test_loader):
    model.eval()
    test_loss, accuracy, accuracy_top_5, total = 0, 0, 0, 0
    
    with torch.no_grad():
        if args.ema > 0:
            ema.store()
            ema.copy_to()
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output, _, _ = model(data)
            if args.rotations:
                output, _ = output
            test_loss += criterion(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            
            # if we want to compute top-5 accuracy
            if top_5:
                preds = output.sort(dim = 1, descending = True)[1][:,:5]
                for i in range(preds.shape[0]):
                    if target[i] in preds[i]:
                        accuracy_top_5 += 1
            # count total number of samples for averaging in the end
            total += target.shape[0]
        if args.ema > 0:
            ema.restore()
    # return results
    model.train()
    
    if args.wandb:
        wandb.log({ "test_loss" : test_loss / total, "test_acc" : accuracy / total, "test_acc_top_5" : accuracy_top_5 / total})

    return { "test_loss" : test_loss / total, "test_acc" : accuracy / total, "test_acc_top_5" : accuracy_top_5 / total}

# function to train a model using args.epochs epochs
# at each args.milestones, learning rate is multiplied by args.gamma
def train_complete(model, loaders, mixup = False):
    global start_time
    start_time = time.time()

    if few_shot:
        train_loader, train_clean, val_loader, novel_loader = loaders
        for i in range(len(few_shot_meta_data["best_val_acc"])):
            few_shot_meta_data["best_val_acc"][i] = 0
    else:
        train_loader, val_loader, test_loader = loaders

    lr = args.lr

    for epoch in range(args.epochs + args.manifold_mixup):

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)

        if (args.cosine and epoch % args.milestones[0] == 0) or epoch == 0:
            if lr < 0:
                optimizer = torch.optim.Adam(model.parameters(), lr = -1 * lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
            if args.cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones[0] * length)
                lr = lr * args.gamma
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(np.array(args.milestones) * length), gamma = args.gamma)

        train_stats = train(model, train_loader, optimizer, (epoch + 1), scheduler, mixup = mixup, mm = epoch >= args.epochs)        
        
        # if args.save_model != "" and not few_shot:
        if args.save_model != "":
            if len(args.devices) == 1:
                torch.save(model.state_dict(), args.save_model)
            else:
                torch.save(model.module.state_dict(), args.save_model)
        
        if (epoch + 1) > args.skip_epochs:
            if few_shot:
                if args.ema > 0:
                    ema.store()
                    ema.copy_to()
                if args.use_supp:
                    res = my_few_shot_eval_word_embed_supp.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
                else:
                    res = my_few_shot_eval_word_embed.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
                if args.ema > 0:
                    ema.restore()
                for i in range(len(args.n_shots)):
                    print("val-{:d}: {:.2f}%, nov-{:d}: {:.2f}% ({:.2f}%) ".format(args.n_shots[i], 100 * res[i][0], args.n_shots[i], 100 * res[i][2], 100 * few_shot_meta_data["best_novel_acc"][i]), end = '')
                    if args.wandb:
                        wandb.log({'epoch':epoch, f'val-{args.n_shots[i]}':res[i][0], f'nov-{args.n_shots[i]}':res[i][2], f'best-nov-{args.n_shots[i]}':few_shot_meta_data["best_novel_acc"][i]})

                print()
            else:
                test_stats = test(model, test_loader)
                if top_5:
                    print("top-1: {:.2f}%, top-5: {:.2f}%".format(100 * test_stats["test_acc"], 100 * test_stats["test_acc_top_5"]))
                else:
                    print("test acc: {:.2f}%".format(100 * test_stats["test_acc"]))

    if args.epochs + args.manifold_mixup <= args.skip_epochs:
        if few_shot:
            if args.ema > 0:
                ema.store()
                ema.copy_to()
            if args.use_supp:
                res = my_few_shot_eval_word_embed_supp.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
            else:
                res = my_few_shot_eval_word_embed.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
                
            if args.ema > 0:
                ema.restore()
        else:
            test_stats = test(model, test_loader)

    if few_shot:
        return few_shot_meta_data
    else:
        return test_stats

### process main arguments
loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
### initialize few-shot meta data
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    if args.dataset.lower() in ["tieredimagenet", "cubfs"]:
        elements_train, elements_val, elements_novel = elements_per_class
    else:
        elements_val, elements_novel = [elements_per_class] * val_classes, [elements_per_class] * novel_classes
        elements_train = None
    print("Dataset contains",num_classes,"base classes,",val_classes,"val classes and",novel_classes,"novel classes.")
    print("Generating runs... ", end='')

    if args.use_supp:
        val_runs = list(zip(*[my_few_shot_eval_word_embed_supp.define_runs(args.n_ways, s, args.n_queries, val_classes, elements_val) for s in args.n_shots]))
        val_run_classes, val_run_indices = val_runs[0], val_runs[1]
        novel_runs = list(zip(*[my_few_shot_eval_word_embed_supp.define_runs(args.n_ways, s, args.n_queries, novel_classes, elements_novel) for s in args.n_shots]))
        novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]
    else:
        val_runs = list(zip(*[my_few_shot_eval_word_embed.define_runs(args.n_ways, s, args.n_queries, val_classes, elements_val) for s in args.n_shots]))
        val_run_classes, val_run_indices = val_runs[0], val_runs[1]
        novel_runs = list(zip(*[my_few_shot_eval_word_embed.define_runs(args.n_ways, s, args.n_queries, novel_classes, elements_novel) for s in args.n_shots]))
        novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]
    print("done.")
    few_shot_meta_data = {
        "elements_train":elements_train,
        "val_run_classes" : val_run_classes,
        "val_run_indices" : val_run_indices,
        "novel_run_classes" : novel_run_classes,
        "novel_run_indices" : novel_run_indices,
        "best_val_acc" : [0] * len(args.n_shots),
        "best_val_acc_ever" : [0] * len(args.n_shots),
        "best_novel_acc" : [0] * len(args.n_shots)
    }

# can be used to compute mean and std on training data, to adjust normalizing factors
if False:
    train_loader, _, _ = loaders
    try:
        for c in range(input_shape[0]):
            print("Mean of canal {:d}: {:f} and std: {:f}".format(c, train_loader.data[:,c,:,:].reshape(train_loader.data[:,c,:,:].shape[0], -1).mean(), train_loader.data[:,c,:,:].reshape(train_loader.data[:,c,:,:].shape[0], -1).std()))
    except:
        pass

### prepare stats
run_stats = {}
if args.output != "":
    f = open(args.output, "a")
    f.write(str(args))
    f.close()

### function to create model
def create_model():
    if args.model.lower() == "resnet18":
        return resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "resnet20":
        return resnet.ResNet20(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "wideresnet":
        return wideresnet.WideResNet(args.feature_maps, input_shape, few_shot, args.rotations, num_classes = num_classes).to(args.device)
    if args.model.lower() == "resnet12":
        print('now')
        return resnet12.ResNet12(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower()[:3] == "mlp":
        return mlp.MLP(args.feature_maps, int(args.model[3:]), input_shape, num_classes, args.rotations, few_shot).to(args.device)
    if args.model.lower() == "s2m2r":
        return s2m2.S2M2R(args.feature_maps, input_shape, args.rotations, num_classes = num_classes).to(args.device)

if args.test_features != "":
    try:
        filenames = eval(args.test_features)
    except:
        filenames = args.test_features
    if isinstance(filenames, str):
        length = filenames.split(",")
        if len(length) > 1:
            filenames = length
        else:
            filenames = [filenames] 
        # filenames = [filenames]   
        print(filenames)     
    test_features = torch.cat([torch.load(fn, map_location=torch.device(args.device)).to(args.dataset_device) for fn in filenames], dim = 2)
    print("Testing features of shape", test_features.shape)
    train_features = test_features[:num_classes]
    val_features = test_features[num_classes:num_classes + val_classes]
    test_features = test_features[num_classes + val_classes:]
    if not args.transductive:
        for i in range(len(args.n_shots)):
            if args.use_supp:
                val_acc, val_conf, test_acc, test_conf = my_few_shot_eval_word_embed_supp.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data)
            else:
                val_acc, val_conf, test_acc, test_conf = my_few_shot_eval_word_embed.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data)
            print("Inductive {:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * test_acc, 100 * test_conf))
    else:
        for i in range(len(args.n_shots)):
            if args.use_supp:
                val_acc, val_conf, test_acc, test_conf = my_few_shot_eval_word_embed_supp.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data, transductive = True)
            else:
                val_acc, val_conf, test_acc, test_conf = my_few_shot_eval_word_embed.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data, transductive = True)
            print("Transductive {:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * test_acc, 100 * test_conf))
    sys.exit()

for i in range(args.runs):

    if not args.quiet:
        print(args)
    if args.wandb:
        wandb.init(project="few-shot", 
            entity=args.wandb, 
            tags=[f'run_{i}', args.dataset], 
            notes=str(vars(args))
            )
        wandb.log({"run": i})
    model = create_model()
    if args.ema > 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)

    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
        model.to(args.device)

    if len(args.devices) > 1:
        model = torch.nn.DataParallel(model, device_ids = args.devices)
    
    if i == 0:
        print("Number of trainable parameters in model is: " + str(np.sum([p.numel() for p in model.parameters()])))

    # training
    test_stats = train_complete(model, loaders, mixup = args.mixup)

    # assemble stats
    for item in test_stats.keys():
        if i == 0:
            run_stats[item] = [test_stats[item].copy() if isinstance(test_stats[item], list) else test_stats[item]]
        else:
            run_stats[item].append(test_stats[item].copy() if isinstance(test_stats[item], list) else test_stats[item])

    # write file output 
    if args.output != "":
        f = open(args.output, "a")
        f.write(", " + str(run_stats))
        f.close()

    # print stats
    print("Run", i + 1, "/", args.runs)
    if few_shot:
        for index in range(len(args.n_shots)):
            stats(np.array(run_stats["best_novel_acc"])[:,index], "{:d}-shot".format(args.n_shots[index]))
            if args.wandb:
                wandb.log({"run": i+1,"test acc {:d}-shot".format(args.n_shots[index]):np.mean(np.array(run_stats["best_novel_acc"])[:,index])})
    else:
        stats(run_stats["test_acc"], "Top-1")
        if top_5:
            stats(run_stats["test_acc_top_5"], "Top-5")

if args.output != "":
    f = open(args.output, "a")
    f.write("\n")
    f.close()
