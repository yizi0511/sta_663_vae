from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt


device = torch.device("cuda")


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 68
        self.w = Variable(torch.randn(5, 1), requires_grad=True).cuda()
        self.a = Variable(torch.randn(68*68), requires_grad=True).cuda()
        self.fc1 = nn.Linear(68*68, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        self.fc31 =  nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        self.fc41 = nn.Linear(68,68)
        self.fc42 = nn.Linear(68,68)
        self.fc43 = nn.Linear(68,68)
        self.fc44 = nn.Linear(68,68)
        self.fc45 = nn.Linear(68,68)
        self.fcy1 = nn.Linear(latent_dim,1)
        self.fcintercept = nn.Linear(68*68, 68*68)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31= F.sigmoid(self.fc31(z))
        h31= F.sigmoid(self.fc41(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = F.sigmoid(self.fc32(z))
        h32 = F.sigmoid(self.fc42(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = F.sigmoid(self.fc33(z))
        h33 = F.sigmoid(self.fc43(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = F.sigmoid(self.fc34(z))
        h34 = F.sigmoid(self.fc44(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = F.sigmoid(self.fc35(z))
        h35 = F.sigmoid(self.fc45(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = torch.cat((h31_out.view(-1,68*68,1), h32_out.view(-1,68*68,1), h33_out.view(-1,68*68,1), h34_out.view(-1,68*68,1), h35_out.view(-1,68*68,1)), 2)
        h30 = torch.bmm(h30, self.w.expand(batch_size,5,1))
        h30 = self.a.expand(batch_size, 68*68) + h30.view(-1,68*68)
        h30 = self.fcintercept(h30)
        return h30.view(-1, 68*68)
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        trait = self.fcy1(mu)
        return recon.view(-1, 68*68), mu, logvar, trait.view(-1,1)



def loss_function(recon_x, that, x, t , mu, logvar):
    BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    NCE = F.mse_loss(that, t, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.01*(BCE + KLD) + NCE




def train(epoch, train_loader_k):
    model.train()
    train_loss = 0
    for batch_idx, (data, trait) in enumerate(train_loader_k):
        data = data.to(device)
        trait = trait.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, traithat = model(data.view(-1,68*68))
        loss = loss_function(recon_batch.view(-1,68*68), traithat.view(-1,1), data.view(-1,68*68), trait.view(-1,1), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader_k.dataset)))


def test(epoch, test_loader_k, test_index):
    model.eval()
    test_loss = 0
    pred_val = torch.zeros(len(test_index))
    test_val = torch.zeros(len(test_index))
    with torch.no_grad():
        for i, (data, trait) in enumerate(test_loader_k):
            data = data.to(device)
            trait = trait.to(device)
            recon_batch, mu, logvar, traithat = model(data)
            test_loss += F.mse_loss(traithat, trait.view(-1,1) ,reduction='sum')
            start = i*batch_size
            end = start + batch_size
            if i == len(test_loader_k) - 1:
                end = len(test_index)
            pred_val[start:end] = traithat.view(-1).cpu()
            test_val[start:end] = trait.view(-1).cpu()
    test_loss /= len(test_index)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, pred_val, test_val




dat_mat = loadmat('../HCP_12_2019/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,0,i] + np.transpose(tensor[:,:,0,i]))
    ith = np.log(ith+1)
    ith = ith[18:86, 18:86]
    ith = ith.reshape(68*68)
    net_data.append(ith)


net_mean = np.mean(net_data, axis=0)
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
phen_dat = loadmat('../HCP_12_2019/HCP_Covariates.mat')
phen_idx = [np.where(phen_dat['allsubject_id'] == ival)[0] for ival in dat_mat['all_id'].squeeze()]
trait_data = phen_dat['cog_measure'][24, phen_idx]
trait_data[np.isnan(trait_data)] = np.nanmean(trait_data)
trait_data = (trait_data-np.mean(trait_data))/np.std(trait_data)
trait_data = torch.from_numpy(trait_data)
trait_data = trait_data.float()
# 4 - Oral Reading Recognition Test
# 6 - Picture Vocabulary Test 
# 24 - Line Orientation: Total number correct
# 26 - Line Orientation: Total positions off for all trials 


batch_size = 5
y = utils.TensorDataset(tensor_y, trait_data) # create your datset
train_loader = utils.DataLoader(y, batch_size) 

nepoch = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for j in range(100):
    train(j, train_loader)
    valid_loss, traithat, trait = test(j, train_loader, np.arange(len(train_loader.dataset)))



torch.cuda.empty_cache()

latent_dim=68
num_elements = len(train_loader.dataset)
num_batches = len(train_loader)
batch_size = train_loader.batch_size
mu_out = torch.zeros(num_elements, latent_dim)
logvar_out = torch.zeros(num_elements,latent_dim)
recon_out = torch.zeros(num_elements,68*68)
x_latent_out = torch.zeros(num_elements, 68)
with torch.no_grad():
    for i, (data, trait) in enumerate(train_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data.to(device)
        trait = trait.to(device)
        recon_batch, mu, logvar, traithat = model(data)
        mu_out[start:end] = mu


np.save('mu.npy', mu_out.detach().numpy())
