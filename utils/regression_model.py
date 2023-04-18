import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from load_regression_data import load_data


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bias = bias
        layer_list = []
        input_dim = self.input_dim
        for i in range(n_layers):
            linear = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias)
            relu = nn.LeakyReLU()
            layer_list.append(linear)
            layer_list.append(relu)
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Linear(in_features=hidden_dim, out_features=self.output_dim, bias=bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = self.head(x)
        return y


class MLPAgent:
    def __init__(self, t_step, max_steps, output_dim, n_layers, hidden_dim, rewards=True, bias=True, dev_str="cpu", lr = 1e-3):
        self.t_step = t_step
        self.max_steps = max_steps

        self.input_dim = t_step*5 + 7
        self.output_dim = output_dim
        self.rewards = rewards

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dev_str = dev_str
        self.lr = lr

        self.model = MLP(self.input_dim, output_dim, n_layers, hidden_dim, bias)
        self.device = torch.device(dev_str)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, eps=1e-7, weight_decay=0.1)

    def format_data(self, data_dict):
        rewards = data_dict["reward_collection"] # shape: 5, 4
        rewards = np.stack(rewards, 0)
        n_trials = len(rewards)
        steps_trial = rewards[0].shape[1]
        perm_obs = data_dict["perm_observations"] # shape: 4
        # bhps = data_dict["max_bhp_inj_collection"] # shape: 4, 4
        sats = data_dict["max_sat_inj_collection"] # shape: 4, 1
        sats = np.stack(sats, 0)[..., 0]
        params = data_dict["parameters_cleaned"] # 13
        # param_names = data_dict["parameter_names"]
        x = np.zeros((n_trials, self.input_dim))
        # well_coords, perms, sats, prior_rewards, global
        y = np.zeros((n_trials, self.output_dim))
        # idx = 0
        x[:, 0:2] = params[:, 11:13]
        if self.t_step != 0:
            x[:, 2:self.t_step*2 + 2] = params[:, 5:5+self.t_step*2]
        perm_start = 2*self.t_step + 2
        x[:, perm_start] = perm_obs[:, -1]
        perm_start += 1
        if self.t_step != 0:
            x[:, perm_start:perm_start + self.t_step] = perm_obs[:, :self.t_step]
        sat_start = perm_start + self.t_step
        if self.t_step != 0:
            x[:, sat_start:sat_start+self.t_step] = sats[:, :self.t_step]

        prior_reward_start = sat_start + self.t_step # TODO
        # if self.t_step > 1:
        #     x[:, prior_reward_start+self.t_step] = rewards
        globals_start = prior_reward_start + self.t_step - 1
        x[:, globals_start:] = params[:, :5]

        if self.rewards: # TODO
            if self.t_step != self.max_steps:
                y = rewards[:, :, self.t_step - 1]
            else:
                y = rewards[:, :, -1]
        # for i in range(n_trials):
        #     for j in range(steps_trial):
        #         x[idx, 0: j+1] = perm_obs[i][j:j+1]
        #         x[idx, 4 : 4 + (j + 1)*2] = params[i, 4:4+(j + 1)*2]
        #         y[idx, :] = rewards[i][:, j]
        #         idx += 1
        return x, y

    def learn(self, epochs, batch_size, data_path, case_name, valid_prop=0.1, save_every=1,
              log_dir="./log/", save_dir="./models/"):
        results = load_data(data_path, case_name)
        x, y = self.format_data(results)

        x -= np.mean(x, axis=0)
        x /= (np.std(x, axis=0) + 1e-6)

        y -= np.mean(y, axis=0)
        y /= (np.std(y, axis=0) + 1e-6)

        n = x.shape[0]
        n_valid = int(np.floor(n*valid_prop))
        n_train = n - n_valid
        n_batches = n_train // batch_size
        data_indices = np.arange(n)
        np.random.shuffle(data_indices)
        valid_indices = data_indices[:n_valid]
        train_indices = data_indices[n_valid:]

        if not os.path.isdir(log_dir):
            print(f"Creating \"{log_dir}\"...")
            os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard_scalars"))

        if save_dir is not None:
            if not os.path.isdir(save_dir):
                print(f"Creating \"{save_dir}\"...")
                os.makedirs(save_dir)

        for epoch in range(epochs):
            np.random.shuffle(train_indices)
            if (epoch%10) == 0:
                print("Starting epoch %i of %i" % (epoch, epochs))
            train_losses = []
            for idx in range(n_batches):
                idxs = train_indices[idx:(idx+1)*batch_size]
                x_batch = torch.from_numpy(x[idxs, :]).to(device=self.device, dtype=torch.float32)
                y_batch = torch.from_numpy(y[idxs, :]).to(device=self.device, dtype=torch.float32)

                y_pred = self.model(x_batch)
                loss = F.huber_loss(y_pred, y_batch)
                train_losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()  # Backpropagate importance-weighted minibatch loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Valid Loss
            with torch.no_grad():
                x_batch = torch.from_numpy(x[valid_indices, :]).to(device=self.device, dtype=torch.float32)
                y_batch = torch.from_numpy(y[valid_indices, :]).to(device=self.device, dtype=torch.float32)
                y_pred = self.model(x_batch)
                valid_loss = F.huber_loss(y_pred, y_batch).item()

            mean_loss = np.mean(train_losses)
            if log_dir is not None:
                tb_writer.add_scalar("Loss/Mean Training", mean_loss, epoch)
                tb_writer.add_scalar("Loss/Validation", valid_loss, epoch)
                tb_writer.flush()
            if save_dir is not None:
                if epoch % save_every == 0:
                    self.save(save_dir, episode=epoch)
        import pdb; pdb.set_trace()
        if save_dir is not None:
            self.save(save_dir)

    def save(self, savedir, episode=None):
        if episode is None:
            epoch = "Final"
            agent_name = "agent_state.tar"
        else:
            epoch = episode
            agent_name = "agent_state-{}.tar".format(episode)

        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(savedir, agent_name))

    def load(self, savedir, episode=None, should_load_optimizer=True):
        if episode is None:
            agent_name = "agent_state.tar"
        else:
            agent_name = "agent_state-{}.tar".format(episode)

        agent_model_path = os.path.join(savedir, agent_name)
        agent_model = torch.load(agent_model_path)
        self.model.load_state_dict(agent_model["model_state_dict"])
        if should_load_optimizer:
            self.optimizer.load_state_dict(agent_model["optimizer_state_dict"])

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

if __name__ == "__main__":
    agent = MLPAgent(3, 4, 5, 2, 16, bias=True)
    data_path = '/home/jmern/Storage/CCS/data/data_regression_temp/'
    case_name = 'eng_geo_no_global_rate02'
    agent.learn(1000, 64, data_path, case_name, save_dir=None)