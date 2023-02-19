###################################################################################################
###################################################################################################
#################################        Adaptive Codebook        #################################
###################################################################################################
###################################################################################################

''' Here the implementation of Learning Multi-Rate Vector Quantization for Remote Deep Inference by
May Malka, Shai Ginzach, and Nir Shlezinger

For further question: maymal@post.bgu.ac.il
'''

###################################################################################################
###################################################################################################
#################################             Imports             #################################
###################################################################################################
###################################################################################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import ssl
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import time
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################################################
###################################################################################################
#################################   Globals & Hyperparameters     #################################
###################################################################################################
###################################################################################################

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
NUM_CLASSES = 100       # Classification of CIFAR100
NUM_EMBED = 256         # Number of vectors in the codebook.

###################################################################################################
###################################################################################################
#################################         Data Arrangment         #################################
###################################################################################################
###################################################################################################

def get_test_transforms():
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
      		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return test_transform

def get_train_transforms():
    transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return transform

train_transform = get_train_transforms()
test_transform = get_test_transforms()

ssl._create_default_https_context = ssl._create_unverified_context
path = "/tmp/cifar100"
trainset = datasets.CIFAR100(root = path, train=True, download=True, transform=train_transform)
testset = datasets.CIFAR100(root = path, train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

###################################################################################################
###################################################################################################
#################################       Adaptive Quantizer        #################################
###################################################################################################
###################################################################################################

class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commintment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

        self.first = True

    def forward(self, inputs, num_vectors, prev_vecs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.d)

        quant_vecs = []
        losses = []

        for num_actives in range(int(np.log2(num_vectors))):
            actives = self.codebook.weight[:pow(2, num_actives + 1)]

            # Calculate distances
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(actives ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, actives.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quant_vecs.append(quantized)

        for num_actives in range(int(np.log2(num_vectors))):
            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quant_vecs[num_actives], inputs.detach())  # commitment loss

                if num_actives == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)  # alignment loss

                elif num_actives == 1:
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)
                    prox_loss = (num_actives * self.lambda_p) * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                                           actives[:pow(2, num_actives + 1) // 2])
                else:
                    e_latent_loss = 0
                    prox_loss = self.lambda_p * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                           actives[:pow(2, num_actives + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.lambda_c * e_latent_loss + prox_loss  # codebook loss

                quant_vecs[num_actives] = inputs + (quant_vecs[num_actives] - inputs).detach()  # gradient copying
                quant_vecs[num_actives] = quant_vecs[num_actives].permute(0, 3, 1, 2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                quant_vecs[num_actives] = quant_vecs[num_actives].permute(0, 3, 1, 2).contiguous()
                cb_loss = 0

            losses.append(cb_loss)

        return quant_vecs, losses, actives


###################################################################################################
###################################################################################################
########################################       Model        #######################################
###################################################################################################
###################################################################################################

class AdapCB_Model(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33, quant=True):
        super(AdapCB_Model, self).__init__()

        self.encoder, self.decoder, self.classifier = self.split_net()
        self.quantizer = AdaptiveVectorQuantizer(num_embeddings, codebook_size, lambda_c, lambda_p)
        self.quant = quant

    def build_model(self, pretrained=True, fine_tune=True):
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        elif not pretrained:
            print('[INFO]: Not loading pre-trained weights')
        inverted_residual_setting=[[1, 16, 1, 1],[6, 24, 2, 1],[6, 32, 3, 1],[6, 64, 4, 2],[6, 96, 3, 1],[6, 160, 3, 1],[6, 320, 1, 1]]
        model = models.mobilenet_v2(pretrained=pretrained, num_classes=1000,width_mult=1,inverted_residual_setting=inverted_residual_setting)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False

        # change the final classification head, it is trainable,
        model.dropout = nn.Dropout(0.2,inplace=True)
        model.fc = nn.Linear(in_features=1280*8*8, out_features=NUM_CLASSES, bias=True)
        return model

    def split_net(self):
        mobilenetv2 = self.build_model()

        encoder = []
        decoder = []
        classifier = []

        res_stop = 5
        for layer_idx, l in enumerate(mobilenetv2.features):
            if layer_idx <= res_stop:
                encoder.append(l)
            else:
                decoder.append(l)

        classifier.append(mobilenetv2.dropout)
        classifier.append(mobilenetv2.fc)

        Encoder = nn.Sequential(*encoder)
        Decoder = nn.Sequential(*decoder)
        Classifier = nn.Sequential(*classifier)
        return Encoder, Decoder, Classifier

    def get_accuracy(self, gt, preds):
        pred_vals = torch.max(preds.data, 1)[1]
        batch_correct = (pred_vals == gt).sum()
        return batch_correct

    def normalize(self,inputs):
        # Calculate the vector's magnitude
        mean = inputs.mean()
        output = inputs - mean
        return output

    def forward(self, inputs, num_active, prev_vecs):
        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer(z_e, num_active, prev_vecs)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                            preds_list[vecs].shape[1]*preds_list[vecs].shape[2]*preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])
        return preds_list, vq_loss, actives, z_e


###################################################################################################
###################################################################################################
##################################       Training Function        #################################
###################################################################################################
###################################################################################################


def train(model, optimizer, num_active, criterion, prev_vecs=None, EPOCHS=20, commitment=0.2):
    start_time = time.time()
    train_losses = []
    encoder_samples = []
    test_losses = []
    val_losses = 0
    stop_criteria = 55
    for epc in range(EPOCHS):
        train_acc = [0] * int(np.log2(num_active))
        val_acc = [0] * int(np.log2(num_active))
        model.train()
        trn_corr = [0] * int(np.log2(num_active))
        losses = 0
        for batch_num, (Train, Labels) in enumerate(trainloader):
            batch_num += 1
            loss_levels = []
            batch = Train.to(device)
            preds_list, vq_loss, curr_vecs, z_e = model(batch, num_active, prev_vecs)
            for q_level in range(len(preds_list)):
                ce_loss = criterion(preds_list[q_level], Labels.to(device))
                level_loss = ce_loss + commitment * vq_loss[q_level]
                loss_levels.append(level_loss)
                train_acc[q_level] += model.get_accuracy(Labels.to(device), preds_list[q_level])

            loss = sum(loss_levels) / len(loss_levels)
            losses += loss.item()

            # Update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_num % 100 == 0:
                print(
                    f'epoch: {epc + 1:2}  batch: {batch_num:2} [{BATCH_SIZE * batch_num:6}/50000]  total loss: {loss.item():10.8f}  \
                time = [{(time.time() - start_time) / 60}] minutes')

        ### Accuracy ###
        loss = losses / batch_num
        train_losses.append(loss)

        model.eval()
        test_losses_val = 0

        with torch.no_grad():

            for b, (X_test, y_test) in enumerate(testloader):
                # Apply the model
                b += 1
                batch = X_test.to(device)
                val_preds, vq_val_loss, _, _ = model(batch, num_active, prev_vecs)
                loss_levels_val = []
                for q_level in range(len(val_preds)):
                    ce_val_loss = criterion(val_preds[q_level], y_test.to(device))
                    level_loss = ce_val_loss + commitment * vq_val_loss[q_level]
                    loss_levels_val.append(level_loss)
                    val_acc[q_level] += model.get_accuracy(y_test.to(device), val_preds[q_level])

                val_loss = sum(loss_levels_val)
                val_losses += val_loss.item()

            test_losses.append(val_losses / b)

        total_train_acc = [100 * (acc.item()) / len(trainset) for acc in train_acc]
        total_val_acc = [100 * (acc.item()) / len(testset) for acc in val_acc]
        # total_train_acc = 100*train_acc/len(trainset)
        # total_val_acc = 100*val_acc/len(testset)
        print(f'Train Models Accuracy at epoch {epc + 1} is {total_train_acc}%')
        print(f'Validation Models Accuracy at epoch {epc + 1} is {total_val_acc}%')
        model.train()

    encoder_samples = z_e
    encoder_samples = encoder_samples.permute(0, 2, 3, 1).contiguous()
    encoder_samples = encoder_samples.view(-1, 2)
    duration = time.time() - start_time
    print(f'Training took: {duration / 3600} hours')
    stop_criteria += 1
    return curr_vecs, encoder_samples


###################################################################################################
###################################################################################################
##################################           Helpers              #################################
###################################################################################################
###################################################################################################

def scatter(array,train_points):
    plt.rcParams["figure.figsize"] = (10,10)
    colors = ['red', 'green', 'blue', 'purple', 'orange','magenta','cyan','yellow']
    for i,level in enumerate(array):
        x = np.array([elem[2] for elem in level.detach().cpu()])
        y = np.array([elem[3] for elem in level.detach().cpu()])
        name =  str(2*pow(2,i)) + 'Bits Vectors'
        train_points_level = train_points[-1]
        train_points_level = train_points_level[:20000]
        train_x_vals = np.array([elem[0] for elem in train_points_level.detach().cpu()])
        train_y_vals = np.array([elem[1] for elem in train_points_level.detach().cpu()])
        plt.scatter(train_x_vals, train_y_vals,s=10, alpha=0.1,label = 'Train Vectors')
        plt.scatter(x, y,s=250, alpha=1,label = name,c=colors[i % 8])

        # Add axis labels and a title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Scatter Plot')
        plt.grid()
        plt.legend(loc='best')
        # Show the plot
        plt.show()

def init_weights(m):
    if type(m) == nn.Embedding:
        torch.nn.init.normal_(m.weight, mean=0, std=1)


###################################################################################################
###################################################################################################
################################      Train Adaptive Codebook       ###############################
###################################################################################################
###################################################################################################


prev_vecs = None                      # Start with empty codebook
criterion = nn.CrossEntropyLoss()

mobilenetv2_ac = AdapCB_Model(num_embeddings=2,
                              codebook_size=NUM_EMBED,
                              quant = True)

mobilenetv2_ac.quantizer.apply(init_weights)  # Initialize weights of codebook from normal distribution (0,1)
mobilenetv2_ac.to(device)
optimizer = torch.optim.Adam(mobilenetv2_ac.parameters(), lr=LEARNING_RATE)

samples_for_scatter = []
vecs_to_save = []
EPOCHS = [10,9,8,7,6,6,6,6]  # Set number of epochs for each training phase.

for level in range(int(np.log2(NUM_EMBED))):
    num_active = pow(2,level+1)        # Number of vectors-to-train in CB
    curr_vecs, encoder_samples = train(mobilenetv2_ac,optimizer,num_active,criterion,prev_vecs,EPOCHS[level])
    samples_for_scatter.append(encoder_samples)
    vecs_to_save.append(curr_vecs)
    prev_vecs = curr_vecs
    scatter(vecs_to_save,samples_for_scatter)
