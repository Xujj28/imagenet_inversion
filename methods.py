import glob
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import random
import torch.optim as optim

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        # print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class CUB(Dataset):
    """CUB200-2011.
    """
    def __init__(self, dataroot="../data", train=True, transform=None):
        # self.root = os.path.join(root, "CUB_200_2011")
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "CUB_200_2011")
        self.train = train
        self.transform = transform

        self.imgs_path, self.targets = self.read_path()
        self.classes = list(set(self.targets))
        self.class_num = len(self.classes)

        self.targets_imgs_dict = dict()
        targets_np = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(target == targets_np)[0]
            self.targets_imgs_dict.update({target: indexes})

    def __getitem__(self, index):
        img_path, target = self.imgs_path[index], self.targets[index]
        img = self.default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

    def read_path(self):
        """Read img, label and split path.
        """
        img_txt_file_path = os.path.join(self.dataroot, 'images.txt')
        img_txt_file = self.txt_loader(img_txt_file_path, is_int=False)
        img_name_list = img_txt_file

        label_txt_file_path = os.path.join(self.dataroot, "image_class_labels.txt")
        label_txt_file = self.txt_loader(label_txt_file_path, is_int=True)
        label_list = list(map(lambda x: x-1, label_txt_file))

        train_test_file_path = os.path.join(self.dataroot, "train_test_split.txt")
        train_test_file = self.txt_loader(train_test_file_path, is_int=True)
        train_test_list = train_test_file

        if self.train:
            train_img_path = [os.path.join(self.dataroot, "images", x) \
                              for i, x in zip(train_test_list, img_name_list) if i]
            train_targets = [x for i, x in zip(train_test_list, label_list) if i]
            imgs_path = train_img_path
            targets = train_targets
        else:
            test_img_path = [os.path.join(self.dataroot, "images", x) \
                             for i, x in zip(train_test_list, img_name_list) if not i]
            test_targets = [x for i, x in zip(train_test_list, label_list) if not i]
            imgs_path = test_img_path
            targets = test_targets

        return imgs_path, targets


    @staticmethod
    def default_loader(path):
        with open(path, "rb") as afile:
            img = Image.open(afile)
            return img.convert("RGB")

    @staticmethod   
    def txt_loader(path, is_int=True):
        """Txt Loader
        Args:
            path:
            is_int: True for labels and split, False for image path
        Returns:
            txt_array: array
        """
        txt_array = []
        with open(path) as afile:
            for line in afile:
                txt = line[:-1].split(" ")[-1]
                if is_int:
                    txt = int(txt)
                txt_array.append(txt)
            return txt_array

class Caltech256(Dataset):
    """Dataset Caltech 256
    Class number: 257
    Train data number: 24582
    Test data number: 6027

    """
    def __init__(self, dataroot, transform=None, train=True):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "Caltech-256")
        self.train = train
        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        # Metadata of dataset
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*'))]
        self.class_num = len(classes)
        self.classes = [i.split('.')[1] for i in classes]
        self.class_to_idx = {i.split('.')[1]: int(i.split('.')[0])-1 for i in classes}
        self.idx_to_class = {int(i.split('.')[0])-1: i.split('.')[1] for i in classes}

        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*', '*'))
        self.targets = [self.class_to_idx[p.split('/')[-2].split('.')[1]] for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """Caltech-256 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class Caltech256_class(Caltech256):
    def __init__(self, dataroot, transform=None, train=True, class_idx=0):
        super(Caltech256_class, self).__init__(dataroot, transform, train)
        self.class_idx = class_idx
        self.indices = [i for (i, j) in enumerate(self.targets) if j == self.class_idx]
        self.targets = [self.targets[i] for i in self.indices]
        self.img_paths = [self.img_paths[i] for i in self.indices]
        
    def __repr__(self):
        repr = """Caltech-256 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}
\tSelected class: {}, {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__(), self.class_idx, self.idx_to_class[self.class_idx])
        return repr

class RobustModel(nn.Module):
    def __init__(self):
        super(RobustModel, self).__init__()
        self.norm = nn.Sequential()
        self.bone = nn.Sequential()
        self.loss_r_feature_layers = []
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        y = self.bone(x_norm)
        return y

    def fv(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        y = x_norm

        for layer in list(self.bone.children())[:-1]:
            y = layer(y)

        # y = list(self.bone.children())[-2](y)
        y = y.view(b, -1)
        return y
    
    def feature_extractor(self):
        classname = self.__class__.__name__
        if classname == 'v11' or classname == 'v16':
            return list(self.bone.features.children())
        elif classname == 'r18' or classname == 'r50':
            return list(self.bone.children())[:-2]

    def fm(self, x, layer_idx):
        classname = self.__class__.__name__
        # print(x.shape)
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        y = x_norm

        cur = 0
        
        for layer in self.feature_extractor():
            y = layer(y)
            if cur == layer_idx:
                break
            cur += 1

        return y

    def classify(self, x):
        b, c, h, w = x.shape
        with torch.no_grad():
            x_norm = self.norm(x)
            y = self.bone(x_norm)
            prob = torch.softmax(y, dim=1)
        return prob

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std
        self.nch = len(mean)

    def forward(self, x):
        y = torch.empty_like(x).to(x.device)
        for i in range(self.nch):
            y[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

        return y

def _init_weight(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.kaiming_normal_(m.weight.data)
            # nn.init.xavier_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1 and len(m.weight.shape)>1:
            # nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.kaiming_normal_(m.weight.data)
            # nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

class r18(RobustModel):
    def __init__(self, class_num, mean, std, init=False):
        super(r18, self).__init__()
        self.class_num = class_num
        self.init = init
        self.norm = Normalization(mean, std)
        self.bone = models.resnet18(pretrained=True)
        self.bone.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)

        if self.init:
            self.bone.apply(_init_weight)

class r50(RobustModel):
    def __init__(self, class_num, mean, std, init=False):
        super(r50, self).__init__()
        self.class_num = class_num
        self.init = init
        self.norm = Normalization(mean, std)
        self.bone = models.resnet50(pretrained=False)
        self.bone.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        self.register_hook()

        if self.init:
            self.bone.apply(_init_weight)
    
    def register_hook(self):
        # print("register_hook")
        for block in self.feature_extractor():
            # print((type(block)))
            if isinstance(block, nn.BatchNorm2d):
                # print("register")
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(block))
            elif isinstance(block, nn.Sequential):
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        # print("register")
                        self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

class v11(RobustModel):
    def __init__(self, class_num, mean, std, init=False):
        super(v11, self).__init__()
        self.class_num = class_num
        self.init = init
        self.norm = Normalization(mean, std)
        self.bone = models.vgg11(pretrained=True)
        self.bone.classifier[6] = nn.Linear(in_features=4096, out_features=class_num, bias=True)

        if self.init:
            self.bone.apply(_init_weight)

class vb11(RobustModel):
    def __init__(self, class_num, mean, std, init=False):
        super(vb11, self).__init__()
        self.class_num = class_num
        self.init = init
        self.norm = Normalization(mean, std)
        self.bone = models.vgg11_bn(pretrained=True)
        self.bone.classifier[6] = nn.Linear(in_features=4096, out_features=class_num, bias=True)

        if self.init:
            self.bone.apply(_init_weight)

class v16(RobustModel):
    def __init__(self, class_num, mean, std, init=False):
        super(v16, self).__init__()
        self.class_num = class_num
        self.init = init
        self.norm = Normalization(mean, std)
        self.bone = models.vgg11(pretrained=True)
        self.bone.classifier[6] = nn.Linear(in_features=4096, out_features=class_num, bias=True)

        if self.init:
            self.bone.apply(_init_weight)

def load_model(name, class_num, mean, std, device):
    if name.find('r18') != -1:
        model = r18(class_num, mean, std)
    elif name.find('r50') != -1:
        model = r50(class_num, mean, std)
    elif name.find('v11') != -1:
        model = v11(class_num, mean, std)
    elif name.find('v16') != -1:
        model = v16(class_num, mean, std)
    elif name.find('vb11') != -1:
        model = vb11(class_num, mean, std)
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    return model

class toy(RobustModel):
    def __init__(self, class_num, mean, std):
        super(v16, self).__init__()
        self.class_num = class_num
        self.norm = Normalization(mean, std)
        self.bone = models.vgg11(pretrained=True)
        self.bone.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.bone.classifier = nn.Sequential(
            nn.Linear(in_features=12544, out_features=10, bias=True)
        )
        self.bone.apply(_init_weight)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=False):
        super(LinfPGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda adv_x, target: nn.functional.cross_entropy(self.model(adv_x), target)

        # Model status
        self.training = self.model.training

    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x+perturbation, 0., 1.) - x

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        atk_loss = self.criterion(adv_x, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * torch.sign(grad)
        else:
            adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=self.device)
        perturbation = self.compute_perturbation(x+perturbation, x)

        return perturbation

    def attack(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        self.model.eval()
        self._model_freeze()
        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)
        for i in range(self.iterations):
            perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation

class L2PGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=5, step=1, iterations=20, criterion=None, random_start=True, targeted=False):
        super(L2PGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda adv_x, target: nn.functional.cross_entropy(self.model(adv_x), target)

        # Model status
        self.training = self.model.training

    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        return perturbation.renorm(p=2, dim=0, maxnorm=self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x+perturbation, 0., 1.) - x

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        atk_loss = self.criterion(adv_x, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        g_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1).view(-1, *([1]*(len(x.shape)-1)))
        grad = grad / (g_norm + 1e-10)
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * torch.sign(grad)
        else:
            adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=self.device)
        perturbation = self.compute_perturbation(x+perturbation, x)

        return perturbation

    def attack(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        self.model.eval()
        self._model_freeze()
        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)
        
        for i in range(self.iterations):
            perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation

class FreeAT(nn.Module):
    def __init__(self, model, optimizer, epsilon=8/255, step=2/255, hops=4, random_start=True):
        super(FreeAT, self).__init__()
        self.model = model
        self.optimizer = optimizer

        self.epsilon = epsilon
        self.step = step
        self.hops = hops

        self.random_start = random_start
    
    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        # return perturbation.renorm(p=2, dim=0, maxnorm=self.epsilon)
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x+perturbation, 0., 1.) - x

        return perturbation

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=x.device)
        perturbation = self.compute_perturbation(x+perturbation, x)

        return perturbation

    def onestep(self, x, perturbation, target):
        adv_x = x + perturbation
        adv_x.requires_grad = True

        y = self.model(adv_x)
        loss = nn.functional.cross_entropy(y, target)
        self.model.zero_grad()
        loss.backward()

        g_adv = adv_x.grad
        adv_x.requires_grad = False

        # Update parameters of model
        self.optimizer.step()

        adv_x = adv_x.detach() + self.step*torch.sign(g_adv)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation, loss.item()

    def update(self, x, target):
        b, c, h, w = x.shape

        perturbation = torch.zeros_like(x).to(x.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)

        # Forward
        for h_idx in range(self.hops):
            perturbation, loss = self.onestep(x, perturbation, target)

        return loss
