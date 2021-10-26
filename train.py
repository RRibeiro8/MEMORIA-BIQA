import argparse
import time
import os
import copy

import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, models, transforms

from koncept_model import model_qa
from data_loader import DataLoader

from scipy import stats
from tqdm import tqdm

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    #x, y = np.float32(x), np.float32(y)
    return stats.pearsonr(x, y)[0]

def srcc(x, y):
    """Spearman rank-order correlation coefficient"""
    #x, y = np.float32(x), np.float32(y)
    return stats.spearmanr(x, y)[0]

def train(dataloaders, dataset_sizes, model, criterion, optimizer, epochs=40):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_srcc = -float('inf')
    best_plcc = -float('inf')

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_plcc = 0
            running_srcc = 0

            pred_scores = []
            gt_scores = []

            # Iterate over data.
            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), desc=phase, total=len(dataloaders[phase])):
                inputs = inputs.detach().to(device)
                labels = labels.detach().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1).float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                pred_scores = pred_scores + outputs.squeeze(1).cpu().tolist()
                gt_scores = gt_scores + labels.cpu().tolist()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_plcc += plcc(pred_scores, gt_scores) * inputs.size(0)
                running_srcc += srcc(pred_scores, gt_scores) * inputs.size(0)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_plcc = running_plcc / dataset_sizes[phase]
            epoch_srcc = running_srcc / dataset_sizes[phase]
            print('{} Loss: {:.4f} Plcc: {:.4f} Srcc: {:.4f}'.format(phase, epoch_loss, epoch_plcc, epoch_srcc))

            # deep copy the model
            if phase == 'val' and epoch_plcc > best_plcc:
                best_plcc = epoch_plcc
                best_srcc = epoch_srcc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Plcc: {:4f}'.format(best_plcc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():

    global args
    print(args)

    model_ft=model_qa(num_classes=1) 
    model_ft=model_ft.to(device)

    # Data loading
    data_loader = DataLoader(batch_size=args.batch_size)
    dataloaders, dataset_sizes = data_loader.get_loader()

    # define loss function (criterion) and optimizer
    mse_loss = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)

    trained_model = train(dataloaders, dataset_sizes, model_ft, mse_loss, optimizer, 10)
    
    torch.save(trained_model.state_dict(),'./models/loggy_biqa.pth')

    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Loggy-BIQA Training')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='Batch size (default: 8)')
    
    args = parser.parse_args()

    main()