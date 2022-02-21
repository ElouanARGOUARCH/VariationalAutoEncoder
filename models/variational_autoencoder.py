import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


class VAE(nn.Module):
    def __init__(self, target_samples,latent_dim, hidden_sizes_encoder, hidden_sizes_decoder):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_samples = target_samples.to(self.device)
        self.p = target_samples.shape[1]
        self.d = latent_dim
        self.hidden_sizes_encoder = hidden_sizes_encoder
        self.hidden_sizes_decoder = hidden_sizes_decoder

        self.encoder = []
        hs = [self.p] + hidden_sizes_encoder + [2 * self.d]
        for h0, h1 in zip(hs, hs[1:]):
            self.encoder.extend([
                nn.Linear(h0, h1),
                nn.ELU(),
            ])
        self.encoder.pop()
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        hs = [self.d] + hidden_sizes_decoder + [2 * self.p]
        for h0, h1 in zip(hs, hs[1:]):
            self.decoder.extend([
                nn.Linear(h0, h1),
                nn.ELU(),
            ])
        self.decoder.pop()
        self.decoder = nn.Sequential(*self.decoder)

        self.optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()), lr=5e-3)
        self.to(self.device)
        self.prior = MultivariateNormal(torch.zeros(self.d).to(self.device), torch.eye(self.d).to(self.device))

    def sample_encoder(self, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        return mu + torch.exp(log_sigma) * torch.randn(list(x.shape)[:-1] + [self.d]).to(self.device)

    def encoder_log_density(self, z, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        return -torch.sum(torch.square((z - mu)/torch.exp(log_sigma))/2 + log_sigma + self.d*torch.log(torch.tensor([2*3.14159265]).to(self.device))/2, dim=-1)

    def sample_decoder(self, z):
        out = self.decoder(z)
        mu, log_sigma = out[..., :self.p], out[..., self.p:]
        return mu + torch.exp(log_sigma) * torch.randn(list(z.shape)[:-1] + [self.p]).to(self.device)

    def DKL_posterior_prior(self,mu, log_sigma):
        return torch.sum(torch.square(mu) + torch.exp(log_sigma)+log_sigma-self.d, dim = -1)/2

    def decoder_log_density(self, x, z):
        out = self.decoder(z)
        mu, log_sigma = out[..., :self.p], out[..., self.p:]
        return -torch.sum(torch.square((x - mu)/torch.exp(log_sigma))/2 + log_sigma + self.p*torch.log(torch.tensor([2*3.14159265]).to(self.device))/2, dim=-1)

    def sample_model(self, num_samples):
        z = self.prior.sample([num_samples])
        return self.sample_decoder(z)

    def sample_proxy(self, x):
        return self.sample_encoder(x)

    def resample_input(self, x):
        z = self.sample_proxy(x)
        return self.sample_model(z)

    def model_log_density(self, x):
        MC_samples = 100
        x = x.unsqueeze(0).repeat(MC_samples, 1, 1)
        z = self.sample_encoder(x)
        return torch.logsumexp(self.decoder_log_density(x,z) + self.prior.log_prob(z) - self.encoder_log_density(z,x) - torch.log(torch.tensor([MC_samples]).to(self.device)), dim = 0)

    def ELBO(self, x):
        out = self.encoder(x)
        mu, log_sigma = out[..., :self.d], out[..., self.d:]
        DKL_values = self.DKL_posterior_prior(mu, log_sigma)
        MC_samples = 5
        x = x.unsqueeze(0).repeat(MC_samples, 1, 1)
        z = self.sample_encoder(x)
        mean_log_ratio = torch.mean(self.decoder_log_density(x,z) - self.encoder_log_density(z,x), dim = 0)
        return  mean_log_ratio - DKL_values

    def loss(self, batch):
        return -torch.mean(self.ELBO(batch))

    def train(self, epochs, batch_size):
        perm = torch.randperm(self.target_samples.shape[0])
        loss_values = [torch.tensor([self.loss(
            self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.target_samples.shape[0])]) for i in
            range(int(self.target_samples.shape[0] / batch_size))]).mean().item()]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(self.target_samples.shape[0])
            for i in range(int(self.target_samples.shape[0] / batch_size) + 1 * (
                    int(self.target_samples.shape[0] / batch_size) != self.target_samples.shape[0] / batch_size)):
                self.optimizer.zero_grad()
                batch_loss = self.loss(
                    self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.target_samples.shape[0])])
                batch_loss.backward()
                self.optimizer.step()

            iteration_loss = torch.tensor([self.loss(
                self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.target_samples.shape[0])]) for i
                in range(int(self.target_samples.shape[0] / batch_size))]).mean().item()
            loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t + 1
                best_parameters = self.state_dict()

        self.load_state_dict(best_parameters)
        self.train_visual(best_loss, best_iteration, loss_values)

    def train_visual(self, best_loss, best_iteration, loss_values):
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        Y1, Y2 = best_loss - (max(loss_values) - best_loss) / 2, max(loss_values) + (max(loss_values) - best_loss) / 4
        ax.set_ylim(Y1, Y2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(loss_values, label='Loss values during training', color='black')
        ax.scatter([best_iteration], [best_loss], color='black', marker='d')
        ax.axvline(x=best_iteration, ymax=(best_loss - best_loss + (max(loss_values) - best_loss) / 2) / (
                    max(loss_values) + (max(loss_values) - best_loss) / 4 - best_loss + (
                        max(loss_values) - best_loss) / 2), color='black', linestyle='--')
        ax.text(0, best_loss - (max(loss_values) - best_loss) / 8,
                'best iteration = ' + str(best_iteration) + '\nbest loss = ' + str(np.round(best_loss, 5)),
                verticalalignment='top', horizontalalignment='left', fontsize=12)
        if len(loss_values) > 30:
            x1, x2 = best_iteration - int(len(loss_values) / 15), min(best_iteration + int(len(loss_values) / 15),
                                                                      len(loss_values) - 1)
            k = len(loss_values) / (2.5 * (x2 - x1 + 1))
            offset = (Y2-Y1)/(6*k)
            y1, y2 = best_loss - offset, best_loss + offset
            axins = zoomed_inset_axes(ax, k, loc='upper right')
            axins.axvline(x=best_iteration, ymax=(best_loss - y1) / (y2-y1), color='black', linestyle='--')
            axins.scatter([best_iteration], [best_loss], color='black', marker='d')
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(loss_values, color='black')
            axins.set_xlim(x1 - .5, x2 + .5)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=3, loc2=4)

    def model_visual(self, num_samples=5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.p == 1 and self.d == 1:
            linspace = 500
            with torch.no_grad():
                tt = torch.linspace(torch.min(self.target_samples), torch.max(self.target_samples), linspace).unsqueeze(
                    1).to(self.device)
                model_density = torch.exp(self.model_log_density(tt))
                model_samples = self.sample_model(num_samples)
                reference_samples = self.prior.sample([num_samples])
                tt_r = torch.linspace(torch.min(reference_samples), torch.max(reference_samples), linspace).unsqueeze(
                    1).to(self.device)
                reference_density = torch.exp(self.prior.log_prob(tt_r))
                proxy_samples = self.sample_proxy(self.target_samples[:num_samples])
            fig = plt.figure(figsize=(28, 16))
            ax1 = fig.add_subplot(221)
            sns.histplot(self.target_samples[:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='red',
                         label="Input Target Samples")
            ax1.legend()

            ax2 = fig.add_subplot(222)
            sns.histplot(proxy_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='orange',
                         label='Proxy samples')
            ax2.legend()

            ax3 = fig.add_subplot(223, sharex=ax1)
            ax3.plot(tt.cpu(), model_density.cpu(), color='blue', label="Output model density")
            sns.histplot(model_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='blue',
                         label='model samples')
            ax3.legend()

            ax4 = fig.add_subplot(224, sharex=ax2)
            ax4.plot(tt_r.cpu(), reference_density.cpu(), color='green', label='reference density')
            sns.histplot(reference_samples[:, 0].cpu(), stat='density', alpha=0.5, bins=125, color='green',
                         label='Reference samples')
            ax4.legend()

        elif (self.p > 1 and self.p <= 5) or (self.d > 1 and self.d <= 5):
            with torch.no_grad():
                target_samples = self.target_samples[:num_samples]
                model_samples = self.sample_model(num_samples)
                reference_samples = self.prior.sample([num_samples])
                proxy_samples = self.sample_proxy(target_samples)
            df_target = pd.DataFrame(target_samples.cpu().numpy())
            df_target['label'] = 'Data'
            df_model = pd.DataFrame(model_samples.cpu().numpy())
            df_model['label'] = 'Model'
            df_x = pd.concat([df_target, df_model])

            df_reference = pd.DataFrame(reference_samples.cpu().numpy())
            df_reference['label'] = 'Reference'
            df_proxy = pd.DataFrame(proxy_samples.cpu().numpy())
            df_proxy['label'] = 'Proxy'
            df_z = pd.concat([df_reference, df_proxy])
            g = sns.PairGrid(df_x, hue="label", height=12 / self.p, palette={'Data': 'red', 'Model': 'blue'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

            g = sns.PairGrid(df_z, hue="label", height=12 / self.d, palette={'Reference': 'green', 'Proxy': 'orange'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

        else:
            dim_displayed = 5
            perm_p = torch.randperm(self.p).to(self.device)
            perm_d = torch.randperm(self.d).to(self.device)
            with torch.no_grad():
                target_samples = self.target_samples[:num_samples]
                model_samples = self.sample_model(num_samples)
                reference_samples = self.prior.sample([num_samples])
                proxy_samples = self.sample_proxy(target_samples)
            df_target = pd.DataFrame(target_samples[:, perm_p][:, :dim_displayed].cpu().numpy())
            df_target['label'] = 'Data'
            df_model = pd.DataFrame(model_samples[:, perm_p][:, :dim_displayed].cpu().numpy())
            df_model['label'] = 'Model'
            df_x = pd.concat([df_target, df_model])

            df_reference = pd.DataFrame(reference_samples[:, perm_d][:, :dim_displayed].cpu().numpy())
            df_reference['label'] = 'Reference'
            df_proxy = pd.DataFrame(proxy_samples[:, perm_d][:, :dim_displayed].cpu().numpy())
            df_proxy['label'] = 'Proxy'
            df_z = pd.concat([df_reference, df_proxy])
            g = sns.PairGrid(df_x, hue="label", height=12 / dim_displayed, palette={'Data': 'red', 'Model': 'blue'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_lower(sns.kdeplot, levels=4)
            g.map_lower(sns.scatterplot, alpha=.05)
            g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')

            g = sns.PairGrid(df_z, hue="label", height=12 / dim_displayed,
                             palette={'Reference': 'green', 'Proxy': 'orange'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_lower(sns.kdeplot, levels=4)
            g.map_lower(sns.scatterplot, alpha=.05)
            g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')
