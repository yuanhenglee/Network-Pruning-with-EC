import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch_pruning as tp

config = {
    'device': 'cuda',
    'n_epochs': 2,
    'lr': 0.002,
    'verbose': True,
    'batch_size': 32,
    'random_seed': 42,
    'image_size': 224
}

transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def validate(model, data_loader, criterion, device):

    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for data in tqdm(data_loader):
            counter += 1
            
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = (
        100. * (valid_running_correct / len(data_loader.dataset))
    )

    return epoch_loss, epoch_acc


def prune_model(model, test_loader):
    
    print('Before Pruning:')
    measure_model_running_time(model, test_loader)

    # 1. setup strategy (L1 Norm)
    strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

    # 2. build dependency graph for vgg11
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224).to(device=config['device']))

    # 3. get a pruning plan from the dependency graph.
    #    use "print(model)" to see the model architecture (to choose which layer you want to prune).
    #    pruning_idxs could be either the percentage or the specific node indexs.
    pruning_idxs = strategy(model.features[0].weight, amount=0.5) 
    # pruning_idxs=[1, 3, 5]
    pruning_plan = DG.get_pruning_plan( model.features[0], tp.prune_conv_out_channel, idxs=pruning_idxs)
    print(pruning_plan)

    # 4. execute this plan after checking (prune the model)
    #    if the plan prunes some channels to zero, 
    #    DG.check_pruning plan will return False.
    if DG.check_pruning_plan(pruning_plan):
        pruning_plan.exec()

    print('After Pruning:')
    measure_model_running_time(model, test_loader)

    return model


def measure_model_running_time(model, test_loader):

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            test_loss, test_acc = validate(
                model, test_loader, nn.CrossEntropyLoss(), config['device']
            )
    
    print('loss: ', test_loss)
    print('accuracy: ', test_acc)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def main():

    model = torchvision.models.vgg11()
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    model.load_state_dict(torch.load('./model_weights/vgg11_MNIST.pt'))
    model = model.to(config['device'])

    test_data = torchvision.datasets.MNIST(root = './dataset/', train=False, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=config['batch_size'], shuffle=True, num_workers = 1)

    model = prune_model(model, test_loader)

if __name__ == '__main__':
    main()