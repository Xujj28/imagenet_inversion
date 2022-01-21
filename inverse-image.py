import torch
import torch.nn as nn
import os
import random
import time
import torch.optim as optim
from dataset import Caltech256, miniImageNet
from methods import L2PGD
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from methods import load_model, lr_cosine_policy, get_image_prior_losses, clip

def get_device_id():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args.local_rank


def save_imgs(t, dst_dir, img_path):
    toPIL = T.ToPILImage()
    bs = t.shape[0]
    for i in range(bs):
        img = toPIL(t[i].detach().cpu())
        dir_name = img_path[i].split("/")[-2]
        file_name = img_path[i].split("/")[-1]
        img_dir = os.path.join(dst_dir, dir_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img.save(os.path.join(img_dir, file_name))


def rebuild_image_fv_bn(image, model, layer_idx):
    with torch.no_grad():
        if len(image.shape) == 3:
            ori_fv = model.fv(image.unsqueeze(0).to(device), layer_idx)
        else:
            ori_fv = model.fv(image.to(device), layer_idx)
    
    def criterion(x, y):
        rnd_fv = model.fv(x)
        return torch.div(torch.norm(rnd_fv-ori_fv, dim=1), torch.norm(ori_fv, dim=1)).mean()\
    
    model.eval()

    if len(image.shape) == 3:
        rand_x = torch.randn_like(image.unsqueeze(0), requires_grad=True, device=device)
    else:
        rand_x = torch.randn_like(image, requires_grad=True, device=device)

    
    iterations_per_layer = 2000
    lr = 0.2
    lr_scheduler = lr_cosine_policy(lr, 100, iterations_per_layer)
    lim_0 = 10
    lim_1 = 10
    var_scale_l2 = 1e-4
    var_scale_l1 = 0.0
    l2_scale = 1e-5
    r_feature = 1e-2
    first_bn_multiplier = 1

    start_time = time.time()
    optimizer = optim.Adam([rand_x], lr=lr, betas=[0.5, 0.9], eps=1e-8)
    for i in range(iterations_per_layer):
        # learning rate scheduling
        lr_scheduler(optimizer, i, i)
        #roll
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(rand_x, shifts=(off1, off2), dims=(2, 3))

        # R_prior losses
        loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

        # l2 loss on images
        loss_l2 = torch.norm(inputs_jit.view(inputs_jit.shape[0], -1), dim=1).mean()

        optimizer.zero_grad()
        main_loss = criterion(inputs_jit, torch.tensor([0]))

        #bn loss
        rescale = [first_bn_multiplier] + [1. for _ in range(len(model.loss_r_feature_layers)-1)]
        loss_r_feature = sum([rescale[idx] * item.r_feature for idx, item in enumerate(model.loss_r_feature_layers)])
        
        loss = main_loss + r_feature * loss_r_feature + var_scale_l2 * loss_var_l2 + var_scale_l1 * loss_var_l1  + l2_scale * loss_l2
        loss.backward()

        optimizer.step()
        rand_x.data = torch.clamp(rand_x.data, 0, 1)
        # rand_x.data = clip(rand_x.data)

    print("inverse --- %s seconds ---" % (time.time() - start_time))
    
    return rand_x



if __name__ == "__main__":
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'


    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    TEST_TRANSFORMS = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    bs = 64

    train_dataset = Caltech256(os.path.join(os.environ["MYDATASETS"], "Caltech-256"), train=True, transform=TEST_TRANSFORMS, inversion=True)
    # train_dataset = miniImageNet(os.path.join(os.environ["PUBLICDATASETS"]), train=True, transform=TEST_TRANSFORMS, inversion=True)
    print(len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=12, sampler=train_sampler, shuffle=False, pin_memory=False)

    r50_pretrain = load_model("r50-pretrained.pth", 257, MEAN, STD, device)
    r50_pretrain = nn.parallel.DistributedDataParallel(r50_pretrain, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)

    dst_dir = "/home/20/junjie/results/fv/inverse_caltech256"
    # dst_dir = "/home/20/junjie/results/miniImageNet_inverse_fm_bn/train"
    for train_data, _, img_path in tqdm(train_loader):
        r_data = rebuild_image_fv_bn(train_data, r50_pretrain.module, 19)
        save_imgs(r_data, dst_dir, img_path)