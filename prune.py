import copy
import importlib
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import VGG, ResNet
from tqdm import tqdm

from utils import measure_global_sparsity


class ModelPruner():
    def __init__(self, model_name: str='resnet18', dataset: str='MNIST', pruning_method: str='by_parameter'):
        assert model_name in ['vgg11', 'resnet18']
        assert dataset in ['MNIST', 'CIFAR10']
        assert pruning_method in ['by_parameter', 'by_channel']
        torch.cuda.empty_cache()

        self.config: dict = self._load_config(dataset)
        self.model: VGG = self._load_model(model_name)
        self.test_loader: DataLoader = self._get_test_loader()
        self.pruning_method:str = pruning_method
        self.cached_model = None
        self.prunable_layer_num = 10 if model_name == 'vgg11' else 20

    @property
    def baseline(self)->dict:
        loss, acc = self._validate(self.model)
        return {'loss': loss, 'acc': acc}

    def prune_model(self, prune_amount_list: list) -> VGG:

        assert(
            len(prune_amount_list) == self.prunable_layer_num
        ), 'The total number of prunable layer is 10.'
        assert(
            all(prune_amount>=0 and prune_amount<1  for prune_amount in prune_amount_list)
        ), 'the prune amount should be in the range of [0,1.0)'

        if self.pruning_method == 'by_parameter':
            return self._prune_model_by_parameter(prune_amount_list)
        elif self.pruning_method == 'by_channel':
            return self._prune_model_by_channel(prune_amount_list)

    def get_fitness_score(self, model: VGG = None):
        
        loss = None
        sparsity = None
        if model == None:
            model = self.cached_model
    
        loss, acc = self._validate(model)
        if self.pruning_method == 'by_parameter':
            sparsity = measure_global_sparsity(model)[-1] * 100
        elif self.pruning_method == 'by_channel':
            ori_size = tp.utils.count_params(self.model)
            sparsity = tp.utils.count_params(model) / ori_size * 100

        print(f'acc: {acc}')
        print(f'loss: {loss}')
        print(f'sparsity: {sparsity}')

        alpha = self.config['loss_alpha']
        return -(alpha*acc + (1-alpha)*sparsity)

    def _prune_model_by_parameter(self, prune_amount_list: list) -> VGG:
        
        model = copy.deepcopy(self.model)

        cnt = 0
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=prune_amount_list[cnt])
                cnt+=1
            elif isinstance(module, torch.nn.Linear) and module.out_features != 10:
                prune.l1_unstructured(module, name="weight", amount=prune_amount_list[cnt])
                cnt+=1

        if self.cached_model is not None:
            del self.cached_model
            torch.cuda.empty_cache()
        self.cached_model = model
        return model


    def _prune_model_by_channel(self, prune_amount_list: list) -> VGG:

        model = copy.deepcopy(self.model)
        strategy = tp.strategy.L1Strategy()
        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs=torch.randn(1,3,self.config['image_size'],
                            self.config['image_size']).to(device=self.config['device']))

        cnt = 0
        for module_name, module in model.named_modules():

            if isinstance(module, torch.nn.Conv2d):
                pruning_idxs = strategy(module.weight, amount=prune_amount_list[cnt])
                pruning_plan = DG.get_pruning_plan( module, tp.prune_conv_out_channel, idxs=pruning_idxs)
                cnt+=1
                pruning_plan.exec()

            elif isinstance(module, torch.nn.Linear) and module.out_features != 10:
                pruning_idxs = strategy(module.weight, amount=prune_amount_list[cnt])
                pruning_plan = DG.get_pruning_plan( module, tp.prune_linear_out_channel, idxs=pruning_idxs)
                cnt+=1
                pruning_plan.exec()

        if self.cached_model is not None:
            del self.cached_model
            torch.cuda.empty_cache()
        self.cached_model = model
        return model

    def _validate(self, model: VGG):

        model.eval()
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        criterion = nn.CrossEntropyLoss()
        times = 0
        if self.config['verbose']:
            progress = tqdm(total=10 if self.config['reduction'] else len(self.test_loader))


        with torch.no_grad():
            for data in self.test_loader:
                if self.config['reduction'] and counter==10:
                    break
                
                features, labels = data
                features = features.to(self.config['device'])
                labels = labels.to(self.config['device'])
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()

                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
                
                counter += 1
                if self.config['verbose']:
                    progress.update(1)

        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = (
            100. * (valid_running_correct / self.config['batch_size'] / counter)
        )

        return epoch_loss, epoch_acc

    def _load_config(self, dataset):
        return getattr(importlib.import_module('config'), f'config_{dataset}'.lower())

    def _load_model(self, model_name:str):
        if model_name == 'vgg11':
            model: VGG = torchvision.models.vgg11()
            model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        elif model_name == 'resnet18':
            model: ResNet = torchvision.models.resnet18()
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
        
        model_path = f"{self.config['model_weights_root_path']}{model_name}_{self.config['dataset']}.pt"
        print(f'Load model: {model_path}...')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.config['device'])))
        model = model.to(self.config['device'])
        return model

    def _get_test_loader(self):
        if self.config['dataset'] == 'CIFAR10':
            transform = transforms.Compose([
                transforms.Resize((self.config['image_size'],self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            try:
                test_data = torchvision.datasets.CIFAR10(root = self.config['dataset_root_path'], train=False, transform=transform)
            except:
                if not os.path.exists(self.config['dataset_root_path']):
                    os.makedirs(self.config['dataset_root_path'])
                test_data = torchvision.datasets.CIFAR10(root = self.config['dataset_root_path'], train=False, transform=transform, download=True)

        elif self.config['dataset'] == 'MNIST':
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize((self.config['image_size'],self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            try:
                test_data = torchvision.datasets.MNIST(root = self.config['dataset_root_path'], train=False, transform=transform)
            except:
                if not os.path.exists(self.config['dataset_root_path']):
                    os.makedirs(self.config['dataset_root_path'])
                test_data = torchvision.datasets.MNIST(root = self.config['dataset_root_path'], train=False, transform=transform, download=True)

        return DataLoader(dataset=test_data, batch_size=self.config['batch_size'], shuffle=True, num_workers = 1)