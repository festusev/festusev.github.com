#!/usr/bin/env python
# coding: utf-8

# In[144]:


import skimage
import skimage.io as skio
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[145]:


import torch.nn as nn


# In[146]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[73]:


def positional_encoding(pos, L):
    PE = []
    for i in range(L):
        PE.append(np.sin((2**i)*np.pi*pos))
        PE.append(np.cos((2**i)*np.pi*pos))
    return np.stack([pos] + PE, axis=-1)


# In[74]:


class NeuralField(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L=L

        self.model = nn.Sequential(
            nn.Linear(L*4 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
    
    def forward(self, pos):
        pe = positional_encoding(pos, self.L)
        pe = torch.tensor(pe, device=device).float()
        pe = pe.reshape(pe.shape[0], -1)
        return self.model(pe)


# In[75]:


class DataLoader:
    def __init__(self, img):
        self.img = img

    def sample(self, N):
        pos = np.random.randint([0, 0], high=self.img.shape[:2], size=(N,2))
        rgb = self.img[pos[:, 0], pos[:, 1]]
        
        return pos/self.img.shape[:2], rgb


# ## Part 2
# NERFs!

# In[76]:


data = np.load(f"imgs/lego_200x200.npz")

# Training images: [100, 200, 200, 3]
images_train = data["images_train"] / 255.0

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = data["c2ws_train"]

# Validation images: 
images_val = data["images_val"] / 255.0

# Cameras for the validation images: [10, 4, 4]
# (camera-to-world transformation matrix): [10, 200, 200, 3]
c2ws_val = data["c2ws_val"]

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float


# In[77]:


K = np.array([[focal, 0, images_train.shape[1]/2], [0, focal, images_train.shape[2]/2], [0, 0, 1]])


# In[81]:


def stack_ones(x):
    return np.concatenate([x, np.ones((1, x.shape[-1]))])


# In[119]:


def transform(c2w, x_c):
    return (c2w @ stack_ones(x_c))[:3]


# In[83]:


def pixel_to_camera(K, uv, s):
    assert uv.shape[0] == 2
    return s*(np.linalg.inv(K)@stack_ones(uv))


# In[84]:


def pixel_to_ray(K, c2w, uv):
    uv = uv.astype(np.float64)
    uv += .5
    ro = c2w @ np.array([0, 0, 0, 1])
    Xc = pixel_to_camera(K, uv.T, 1)

    if len(c2w.shape) > 2:
        Xw = np.array([transform(c2w[i], Xc[:, i:i+1]) for i in range(c2w.shape[0])])[:, :3, 0]
        ro = ro[:, :3]
        rd = Xw - ro    
    else:
        Xw = transform(c2w, Xc) 
        ro = ro[:3]
        rd = Xw.T - ro
        
    return ro, rd/np.linalg.norm(rd, axis=-1)[:, np.newaxis]


# In[85]:


all_coords = np.meshgrid(np.arange(200), np.arange(200))
all_coords[0] = all_coords[0].flatten()
all_coords[1] = all_coords[1].flatten()
all_coords = np.array(all_coords).T


# In[86]:


class RaysData:
    def __init__(self, images, K, c2ws):
        self.images = images
        self.K = K
        self.c2ws = c2ws

    def get_rays(self, image_index):
        all_coords = np.meshgrid(np.arange(self.images.shape[1]), np.arange(self.images.shape[2]))
        all_coords[0] = all_coords[0].flatten()
        all_coords[1] = all_coords[1].flatten()
        all_coords = np.array(all_coords).T
        
        c2ws = self.c2ws[image_index]
        return *pixel_to_ray(K, c2ws, all_coords), self.images[image_index]

    def sample_rays(self, N):
        idx = np.random.randint([0, 0, 0], high=self.images.shape[:3], size=(N, 3))
        c2ws = self.c2ws[idx[:, 0]] # First index is which image
        return *pixel_to_ray(K, c2ws, idx[:, 1:]), self.images[idx[:, 0], idx[:, 2], idx[:, 1]]


# In[87]:


def sample_along_rays(rays_o, rays_d, high=6.0, low = 2.0, num_samples=64, perturb=True):
    n = rays_o.shape[0]

    t_width = (high - low)/num_samples
    t = np.linspace(low, high, num_samples + 1)[:-1]
    
    if perturb:
        t = t + (np.random.rand(t.shape[0]) * t_width)
        
    t = t.reshape(1, -1, 1)
    ray_d_expanded = rays_d[:, np.newaxis, :]
    points_along_ray_d = t * ray_d_expanded

    if len(rays_o.shape) > 1:
        ray_o_expanded = rays_o[:, np.newaxis, :]
        points = ray_o_expanded + points_along_ray_d
    else:
        points = rays_o + points_along_ray_d

    return points, t


# ### 2.4 NERF Training

# In[196]:


import datetime
import os

# Get the current time
now = datetime.datetime.now()

# Format the time
run_id = now.strftime('%Y_%m_%d-%H_%M_%S')
os.mkdir(f"runs/{run_id}")


# In[199]:


os.mkdir(f"runs/{run_id}/val_imgs/")
os.mkdir(f"runs/{run_id}/models/")

# In[88]:


class NERF(nn.Module):
    def __init__(self, L_coord=10, L_ray=4):
        super().__init__()
        self.L_coord = L_coord
        self.L_ray = L_ray

        self.coord_inp_size = 3*(self.L_coord*2 + 1)
        self.rd_inp_size = 3*(self.L_ray*2 + 1)

        self.pos_model1 = nn.Sequential(
            nn.Linear(self.coord_inp_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.pos_model2 = nn.Sequential(
            nn.Linear(256 + self.coord_inp_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.density_model = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.rgb_model1 = nn.Linear(256, 256)

        self.rgb_model2 = nn.Sequential(
            nn.Linear(256 + self.rd_inp_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
    def forward(self, pos, rd):        
        encoded_pos = positional_encoding(pos, self.L_coord)
        encoded_pos = torch.tensor(encoded_pos, device=device).float()
        encoded_pos = encoded_pos.reshape(*encoded_pos.shape[:2], -1)

        encoded_ray = positional_encoding(rd, self.L_ray)
        encoded_ray = torch.tensor(encoded_ray, device=device).float()
        encoded_ray = encoded_ray.reshape(encoded_ray.shape[0], 1, -1)
        tiled_ray = torch.tile(encoded_ray, (1, pos.shape[1], 1))
        
        p1 = self.pos_model1(encoded_pos) #TODO Add residual
        z1 = torch.cat([p1, encoded_pos], axis=-1)
        p2 = self.pos_model2(z1)

        density = self.density_model(p2)

        r1 = self.rgb_model1(p2) # TODO actually add in rd
        z2 = torch.cat([r1, tiled_ray], axis=-1)
        rgb = self.rgb_model2(z2)
        
        return density, rgb


# In[89]:


def get_volrend_weights(sigmas, t, device=device):
    t = torch.tensor(t, device=device)
    t = t[:, 1:, :] - t[:, :-1, :]
    t = torch.cat([t, torch.tensor([[[torch.mean(t).item()]]], device=device)], dim=1)
    # t = torch.cat([torch.tensor([[[0]]], device=device), torch.tensor(t,device=device)], dim=1)
    
    T = torch.exp(-torch.cumsum(sigmas*t, dim=1))
    T = torch.cat([torch.ones(T.shape[0], 1, T.shape[-1], device=device), T], axis=1) # Add in 1 to the start
    return T[:, :-1, :]*(1-torch.exp(-sigmas*t)), T

def volrend(sigmas, rgbs, step_size, device=device, bg_color=[0, 0, 0]):
    w, T = get_volrend_weights(sigmas, step_size, device=device)

    C_r = w*rgbs
    C_r = torch.cat([C_r, (T[:, -1:, :]*torch.tensor(bg_color, device=device))], dim=1)
    C_r = torch.sum(C_r, axis=1)
    return C_r
                         
def volrend_old(sigmas, rgbs, step_size, device=device, bg_color=[0, 0, 0]):
    T = torch.exp(-torch.cumsum(sigmas*step_size, dim=1))
    T = torch.cat([torch.ones(T.shape[0], 1, T.shape[-1], device=device), T], axis=1) # Add in 1 to the start

    C_r = T[:, :-1, :]*((1-torch.exp(-sigmas*step_size))*rgbs)
    C_r = torch.cat([C_r, (T[:, -1:, :]*torch.tensor(bg_color, device=device))], dim=1)
    C_r = torch.sum(C_r, axis=1)
    return C_r

# In[113]:


def predict_image(model, dataset, index, bg_color=[0, 0, 0]):
    with torch.no_grad():
        rays_o, rays_d, img = dataset.get_rays(index)
        points, t = sample_along_rays(rays_o, rays_d, perturb=False)
    
        density_pred, rgb_pred = model.forward(points, rays_d)
        C_r = volrend(density_pred, rgb_pred, t, bg_color=bg_color) # volrend_old(density_pred, rgb_pred, (6.0 - 2.0) / 32)
    
        val_loss = mse(torch.tensor(img.reshape(-1, 3), device=device).float(), C_r)
        full_val_loss.append(val_loss)
        
        reconstructed_shell = np.zeros(img.shape)
        reconstructed_shell[all_coords[:, 1], all_coords[:, 0]] = C_r.cpu().detach().numpy()
        return reconstructed_shell


# In[114]:


dataset = RaysData(images_train, K, c2ws_train)
model = NERF().to(device)


# In[115]:


val_dataset = RaysData(images_val, K, c2ws_val)


# In[116]:


optim = torch.optim.Adam(model.parameters(), lr=5e-4)
batch_size = 10000
mse = nn.MSELoss()
training_loss = []
training_imgs = [] # Taken every 200 epochs
full_val_loss = [] # Taken every 200 epochs
high = 6.0
low = 2.0
num_samples = 64


# In[210]:
def standardize_img(img):
    return (img*255).astype(np.uint8)


def train_loop(num_epochs):
    pbar = tqdm(range(len(training_loss), len(training_loss) + num_epochs))
    for epoch in pbar:
        rays_o, rays_d, pixels = dataset.sample_rays(batch_size)
        points, t = sample_along_rays(rays_o, rays_d, high=high, low=low, num_samples=num_samples, perturb=True)
    
        density_pred, rgb_pred = model.forward(points, rays_d)
        C_r = volrend(density_pred, rgb_pred, t)

        loss = mse(torch.tensor(pixels, device=device, dtype=torch.float64), C_r)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        training_loss.append(loss.detach().cpu().numpy().item())

        if epoch%200 == 0:
            torch.save(model.state_dict(), f"runs/{run_id}/models/nerf_{epoch}")

            val_loss = 0
            for i in range(images_val.shape[0]):
                val_img = images_val[i]
                val_img_pred = predict_image(model, val_dataset, i) 

                val_loss += mse(torch.tensor(val_img_pred), torch.tensor(val_img))
                if i==0:
                    # skio.imshow(val_img_pred)
                    # plt.show()
                    training_imgs.append(val_img_pred)
                    skio.imsave(f"runs/{run_id}/val_imgs/{epoch}.jpg", standardize_img(val_img_pred))
                    
            full_val_loss.append((val_loss/images_val.shape[0]).cpu().item())
            
        pbar.set_description(f"Epoch {epoch}: training loss {training_loss[-1]}")
    
    save_data = {"training_loss": training_loss, "full_val_loss": full_val_loss}
    np.save(f"runs/{run_id}/data.npy", save_data)


train_loop(10000)
# In[201]:



# In[197]:


test_dataset = RaysData(np.zeros((c2ws_test.shape[0], 200, 200, 3)), K, c2ws_test)


def make_gif(fname="movie.gif", bg_color=[0, 0, 0]):
    frames = []
    with torch.no_grad():
        for i in range(60):
            img_pred = predict_image(model, test_dataset, i, bg_color=bg_color)
            frames.append(img_pred)

    standardized_frames = [standardize_img(frame) for frame in frames]

    import imageio
    imageio.mimsave(f'runs/{run_id}/{fname}', standardized_frames)

make_gif()
import pdb; pdb.set_trace()

