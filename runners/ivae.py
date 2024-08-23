import time

import torch
import torch.nn.functional as F
import wandb
from data import SyntheticDataset
from metrics import mean_corr_coef as mcc
from models import cleanIVAE, cleanVAE, Discriminator, permute_dims
from torch import optim
from torch.utils.data import DataLoader


def runner(args, config, verbose=False):
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))

    factor = config.gamma > 0

    if wandb.run:
        data_path = config.data_path
    else:
        data_path = args.data_path

    if config.num_segments == -1:
        print(f"Got {config.num_segments=}, overriding to 2*source_dim + 1, i.e., {config.source_dim * 2 + 1}")
        num_segments = config.source_dim * 2 + 1
    else:
        num_segments = config.num_segments

    if config.data_dim == -1:
        print(f"Got {config.data_dim=}, overriding source_dim, i.e., {config.source_dim}")
        data_dim = config.source_dim
    else:
        data_dim = config.data_dim

    dset_train = SyntheticDataset(data_path, config.num_per_segment, num_segments, config.source_dim, data_dim,
                                  config.nl, config.data_seed, config.prior,
                                  config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor,
                                  use_sem=config.use_sem, one_hot_labels=config.one_hot_labels, chain=config.chain,
                                  staircase=config.staircase, dag_mask_prob = config.dag_mask_prob)

    # use different seed for validation set
    # ensure that both have the same adjacency matrix
    dset_val = SyntheticDataset(data_path, config.num_per_segment, num_segments, config.source_dim, data_dim, config.nl,
                                config.data_seed + 1752, config.prior,
                                config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor,
                                use_sem=config.use_sem, one_hot_labels=config.one_hot_labels, chain=config.chain,
                                staircase=config.staircase, adjacency=dset_train.adjacency, dag_mask_prob = config.dag_mask_prob)

    d_data, d_latent, d_aux = dset_train.get_dims()

    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    source_dim_train = DataLoader(dset_train, batch_size=config.batch_size, shuffle=True, drop_last=True,
                                  **loader_params)
    source_dim_val = DataLoader(dset_val, batch_size=config.batch_size, shuffle=False, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []

    if config.ica:
        model = cleanIVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim,
                          n_layers=config.n_layers, activation=config.activation, slope=.1, use_strnn=config.use_strnn,
                          separate_aux=config.separate_aux, residual_aux=config.residual_aux, use_chain=config.chain,
                          strnn_width=config.strnn_width, strnn_layers=config.strnn_layers,
                          aux_net_layers=config.aux_net_layers, ignore_u=config.ignore_u,
                          cond_strnn=config.cond_strnn, adjacency=dset_train.adjacency if config.strnn_adjacency_override is True else None).to(config.device)
    else:
        model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
                         n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)

    if factor:
        D = Discriminator(d_latent).to(config.device)
        optim_D = optim.Adam(D.parameters(), lr=config.lr,
                             betas=(.5, .9))

    loss_hist = []
    perf_hist = []
    for epoch in range(1, config.epochs + 1):
        model.train()

        if config.anneal:
            a = config.a
            d = config.d
            b = config.b
            c = 0
            if epoch > config.epochs / 1.6:
                b = 1
                c = 1
                d = 1
                a = 2 * config.a
        else:
            a = config.a
            b = config.b
            c = config.c
            d = config.d

        train_loss = 0
        train_perf = 0
        for i, data in enumerate(source_dim_train):
            if not factor:
                x, u, s_true = data
            else:
                x, x2, u, s_true = data
            x, u = x.to(config.device), u.to(config.device)
            optimizer.zero_grad()
            loss, z = model.elbo(x, u, len(dset_train), a=a, b=b, c=c, d=d)
            if factor:
                D_z = D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                loss += config.gamma * vae_tc_loss

            loss.backward(retain_graph=factor)

            train_loss += loss.item()
            try:
                perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
            except:
                perf = 0
            train_perf += perf

            optimizer.step()

            if factor:
                ones = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
                zeros = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
                x_true2 = x2.to(config.device)
                _, _, _, z_prime = model(x_true2)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = D(z_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                optim_D.zero_grad()
                D_tc_loss.backward()
                optim_D.step()

        train_perf /= len(source_dim_train)
        perf_hist.append(train_perf)
        train_loss /= len(source_dim_train)
        loss_hist.append(train_loss)
        if verbose:
            print('==> Epoch {}/{}:\ttrain loss: {:.6f}\ttrain perf: {:.6f}'.format(epoch, config.epochs, train_loss,
                                                                                train_perf))

        # if torch.isnan(train_loss):
        #     break

        if wandb.run:
            wandb.log({'train_loss': train_loss, 'train_mcc': train_perf})

        if not config.no_scheduler:
            scheduler.step(train_loss)

        if epoch % config.val_freq == 0 or epoch == config.epochs:
            model.eval()
            val_loss = 0
            val_perf = 0
            for i, data in enumerate(source_dim_val):
                if not factor:
                    x, u, s_true = data
                else:
                    x, x2, u, s_true = data
                x, u = x.to(config.device), u.to(config.device)
                loss, z = model.elbo(x, u, len(dset_train), a=a, b=b, c=c, d=d)
                if factor:
                    D_z = D(z)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                    loss += config.gamma * vae_tc_loss

                val_loss += loss.item()
                try:
                    perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
                except:
                    perf = 0
                val_perf += perf

            val_perf /= len(source_dim_val)
            val_loss /= len(source_dim_val)
            if verbose:
                print('==> Epoch {}/{}:\tval loss: {:.6f}\tval perf: {:.6f}'.format(epoch, config.epochs, val_loss,
                                                                                val_perf))
            if wandb.run:
                wandb.log({'val_loss': val_loss, 'val_mcc': val_perf})
    print('\ntotal runtime: {}'.format(time.time() - st))

    # evaluate perf on full dataset
    Xt, Ut, St = dset_train.x.to(config.device), dset_train.u.to(config.device), dset_train.s
    if config.ica:
        _, _, _, s, _ = model(Xt, Ut)
    else:
        _, _, _, s = model(Xt)
    full_perf = mcc(dset_train.s.numpy(), s.cpu().detach().numpy())
    perfs.append(full_perf)
    loss_hists.append(loss_hist)
    perf_hists.append(perf_hist)

    return perfs, loss_hists, perf_hists
