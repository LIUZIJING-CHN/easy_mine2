import torch
import numpy as np
import os
from torchmetrics.functional import pairwise_euclidean_distance, pairwise_cosine_similarity
from args import *
from utils import *
from einops import rearrange

n_runs = args.n_runs
batch_few_shot_runs = args.batch_fs
assert(n_runs % batch_few_shot_runs == 0)


memory_path = args.memorybank_path
memory_tag = False
if os.path.exists(memory_path):
    memory_tag = True
    memory = torch.load(memory_path).to(args.device)
    print(memory.ndim)
    if memory.ndim == 3:
        memory = rearrange(memory, 'N C M -> (N M) C')
    print(memory.shape)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    
    # print(run_classes)
    
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)
    return res

def ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            
            #####
            # print(means.shape)
            means = means.squeeze()
            cos_similarity = pairwise_cosine_similarity(means.clone(), memory.clone())
            alpha = 1
    
            ## 1
            d_sq, _index = torch.topk(cos_similarity, k=10, dim=1) # 形状都是K*5的
            # print(d_sq)
            tag = torch.ones_like(d_sq)*1
            d_sq_tag = (d_sq>tag)
            d_sq = d_sq*d_sq_tag
            # print(_index)
            # print('feature: ', features.shape)
            # print('d_sq: ',d_sq.shape)
            selected_query_feat = memory[_index,...] # 选出来的特征，1 1 640
            # print('selected: ',selected_query_feat.shape)
            # tmp1 = means + torch.sum((d_sq**alpha).unsqueeze(-1)*selected_query_feat,dim=1)
            # # print('tmp1: ',tmp1.shape)
            # tmp2 = 1+torch.sum(d_sq**alpha,dim=1)
            # # print('tmp2: ',tmp2.shape)
            # means = (tmp1/tmp2.unsqueeze(-1)).unsqueeze(0)
            means = means * 0.9 + 0.1 * torch.sum((d_sq**alpha).unsqueeze(-1)*selected_query_feat,dim=1)
            # print('update: ',features.shape)
            
            # d_sq, _index = torch.topk(cos_similarity, k=18, dim=1) # 形状都是K*5的
            # selected_query_feat = memory[_index,...] # 选出来的特征，1 1 640
            # tmp1 = features + torch.sum((d_sq**alpha).unsqueeze(-1)*selected_query_feat,dim=1)
            # tmp2 = 1+torch.sum(d_sq**alpha,dim=1)
            # features = (tmp1/tmp2.unsqueeze(-1))
            
            #######
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            
            # print((winners == targets).float().mean(dim = 1).mean(dim = 1))
            
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
            
        return stats(scores, "")
    
    
def ncm_old(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            
            print((winners == targets).float().mean(dim = 1).mean(dim = 1))
            
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
            
        return stats(scores, "")

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                if cosine:
                    new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                if cosine:
                    means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def kmeans(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(500):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                allocation = new_allocation
                means = (runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])) * args.n_queries) / runs.shape[2]
            winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def softkmeans(train_features, features, run_classes, run_indices, n_shots, transductive_temperature_softkmeans=args.transductive_temperature_softkmeans, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            runs = postprocess(runs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(30):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                means = torch.sum(runs[:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3]))
                means = means/(n_shots+soft_allocations.sum(dim = 1).reshape(batch_few_shot_runs, -1, 1))
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(batch_few_shot_runs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def get_features(model, attribute_model, loader, no_trainset_tag, n_aug = args.sample_aug):
    model.eval()
    attribute_model.eval()
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):        
            with torch.no_grad():
                # print(batch_idx)
                data, target = data.to(args.device), target.to(args.device)
                _, features = model(data)
                # print('feature1')
                ####
                # features, _ = attribute_model(features)
                # print('feature2')
                if memory_tag and no_trainset_tag and args.save_features:
                    _, C, W, H = features.shape
                    feature_list = []
                    
                    # 加入替换特征
                    for i in range(features.shape[0]):
                        feature_item = features[i].view(C, -1).permute(1, 0)
                        #### 对比特征和库的特征
                        memory_simi, memory_simi_ind = pairwise_euclidean_distance(feature_item.clone(), memory.clone()).min(dim=1)
                        #### 筛选相似度最高的点位
                        _, min_simi_ind = memory_simi.topk(k=15, dim=0, largest=False)
                        # _, min_simi_ind = memory_simi.topk(k=5, dim=0)
                        
                        feature_item_list = []
                        feature_item_avg = F.avg_pool2d(features[i], features[i].shape[2]).view(-1, C)
                        feature_item_list.append(feature_item_avg)  ##加入原有的特征
                        for ind in range(min_simi_ind.shape[0]):    #加入替换特征
                            replace_feature = feature_item
                            replace_feature[min_simi_ind[ind]] = memory[memory_simi_ind[min_simi_ind[ind]]]
                            replace_feature = replace_feature.permute(1, 0)
                            replace_feature = replace_feature.view(C, W, H).unsqueeze(0)
                            replace_feature = F.avg_pool2d(replace_feature, replace_feature.shape[2]).view(-1, C)
                            # print(replace_feature.shape)
                            feature_item_list.append(replace_feature)
                        for ind_first in range(min_simi_ind.shape[0]-1):
                            replace_feature = feature_item
                            replace_feature[min_simi_ind[ind_first]] = memory[memory_simi_ind[min_simi_ind[ind_first]]]
                            for ind_second in range(ind_first+1, min_simi_ind.shape[0]):
                                replace_feature_new = replace_feature
                                replace_feature_new[min_simi_ind[ind_second]] = memory[memory_simi_ind[min_simi_ind[ind_second]]]
                                replace_feature_new = replace_feature_new.permute(1, 0)
                                replace_feature_new = replace_feature_new.view(C, W, H).unsqueeze(0)
                                # print(replace_feature.shape)
                                replace_feature_new = F.avg_pool2d(replace_feature_new, replace_feature_new.shape[2]).view(-1, C)
                                # print(replace_feature.shape)
                                feature_item_list.append(replace_feature_new)
                        
                        feature_item_new = torch.cat(feature_item_list, dim=0).mean(dim=0, keepdim=True)
                        # print(feature_item_new.shape)
                        feature_list.append(feature_item_new)
                    features = torch.cat(feature_list, dim=0)

                else:    
                    features = F.avg_pool2d(features, features.shape[2])
                    features = features.view(features.size(0), -1)
                    # print(features.shape)
                    # print(batch_idx)
                    
                ####
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = int(max_offset - offset + 1)
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)

def update_few_shot_meta_data(model, attribute_model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, attribute_model, train_clean, no_trainset_tag=False)
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, attribute_model, val_loader, no_trainset_tag=True)
    novel_features = get_features(model, attribute_model, novel_loader, no_trainset_tag=True)

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model, attribute_model=attribute_model))

    return res

def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, attribute_model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                    
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
                    
            if args.save_attri_model != "":
                if len(args.devices) == 1:
                    torch.save(attribute_model.state_dict(), args.save_attri_model + str(args.n_shots[index]))
                    
                else:
                    torch.save(attribute_model.module.state_dict(), args.save_attri_model + str(args.n_shots[index]))
                    
            if args.save_features != "":
                torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf

print("eval_few_shot, ", end='')
