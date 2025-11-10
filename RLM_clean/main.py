import os
import sys
import time
import copy
sys.path.append('~/rhm-training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import math
import random

import functools
import argparse

import datasets, models
import init, measures
import matplotlib.pyplot as plt


def run(config):
    # number of input positions for the tree level
    config.input_size = 2**config.num_layers

    if not hasattr(config, 'max_epochs'):
        config.max_epochs = 1

    # only test set is fixed; training will be sampled online
    config.num_data = config.test_size

    # fixed test set
    rlm_test = datasets.RLM(
        v=config.num_features,
        L=config.num_layers,
        beta=config.beta,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,
        num_data=config.test_size,
        probs=None,
        transform=None
    )
    inputs_test = rlm_test.trees[config.num_layers]
    targets_test = rlm_test.trees[0]
    _, test_loader = init.init_data(inputs_test, targets_test, config)

    # model + training setup
    model = init.init_model(config)
    model0 = copy.deepcopy(model)

    criterion, optimizer, scheduler = init.init_training(model, config)
    print_ckpts, save_ckpts = init.init_loglinckpt(config.print_freq, config.max_iters, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    step = 0
    dynamics, best = init.init_output(model, criterion, None, test_loader, config)

    # online RLM for training (different sample seed to decorrelate from test)
    rlm_train = datasets.RLM(
        v=config.num_features,
        L=config.num_layers,
        beta=config.beta,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample + 1,
        num_data=None,    # no fixed dataset in memory
        probs=None,
        transform=None
    )

    if config.checkpoints:
        torch.save(
            {'config': config, 'rules': rlm_train.M},
            f"{config.outname}_config.pt"
        )
        output = {
            'model': copy.deepcopy(model.state_dict()),
            'state': dynamics[-1],
            'step': step
        }
        torch.save(
            output,
            f"{config.outname}_t{step}.pt"
        )

    running_loss = 0.0
    test_loss = dynamics[-1]['testloss']
    test_acc = dynamics[-1]['testacc']
    train_loader = None  # kept only so measure_train logic does not break

    print(config.device)
    print(f"Training for {config.max_iters} steps in online regime")

    while step < config.max_iters:
        # sample fresh batch from RLM grammar
        inputs_raw, targets_raw = rlm_train.sample_batch(
            batch_size=config.batch_size,
            L=config.num_layers,
            mode=config.mode,
            num_tokens=config.num_tokens,
            num_features=config.num_features
        )

        inputs_raw = inputs_raw.to(config.device)
        targets_raw = targets_raw.to(config.device)

        # same target logic as in init.init_data for 'last' mode
        if config.mode == 'class':
            targets = targets_raw
            inputs = inputs_raw
        elif config.mode == 'last':
            targets = torch.clone(inputs_raw[:, -1])
            inputs = inputs_raw
        elif config.mode == 'auto':
            raise NotImplementedError("online 'auto' mode not implemented")
        else:
            raise ValueError(f"invalid mode {config.mode}")

        # apply same encoding pipeline as offline
        inputs = init.transform_inputs(inputs, config)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        loss = loss / config.accumulation
        loss.backward()

        if (step + 1) % config.accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        if step == print_ckpt:
            test_loss, test_acc = measures.test(model, criterion, test_loader, config.device)
            if test_loss < best['loss']:
                best['step'] = step
                best['loss'] = test_loss
                best['model'] = copy.deepcopy(model.state_dict())

            print(
                'step : ', step,
                '\t running loss: {:06.4f}'.format(running_loss / step),
                ', test loss: {:06.4f}'.format(test_loss)
            )
            print_ckpt = next(print_ckpts)

            if step >= save_ckpt:
                print(f'Checkpoint at step {step}, saving data ...')
                save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                if config.measure_train and train_loader is not None:
                    train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                    save_dict['trainloss'] = train_loss
                    save_dict['trainacc'] = train_acc
                dynamics.append(save_dict)

                if config.checkpoints:
                    output = {
                        'model': copy.deepcopy(model.state_dict()),
                        'state': dynamics[-1],
                        'step': step
                    }
                    torch.save(
                        output,
                        f"{config.outname}_t{step}.pt"
                    )
                else:
                    output = {
                        'entropy': rlm_train.entropy,
                        'marginal': rlm_train.marginal,
                        'dynamics': dynamics,
                        'step': step
                    }
                    torch.save(
                        {'config': config, 'output': output},
                        f"{config.outname}.pt"
                    )
                save_ckpt = next(save_ckpts)

        if (running_loss / step) <= config.loss_threshold:
            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            if config.measure_train and train_loader is not None:
                train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                save_dict['trainloss'] = train_loss
                save_dict['trainacc'] = train_acc
            dynamics.append(save_dict)

            if config.checkpoints:
                output = {
                    'model': copy.deepcopy(model.state_dict()),
                    'state': dynamics[-1],
                    'step': step
                }
                torch.save(
                    output,
                    f"{config.outname}_t{step}.pt"
                )
            else:
                output = {
                    'entropy': rlm_train.entropy,
                    'marginal': rlm_train.marginal,
                    'dynamics': dynamics,
                    'step': step
                }
                torch.save(
                    {'config': config, 'output': output},
                    f"{config.outname}.pt"
                )
            break

    print(rlm_train.entropy)
    print(rlm_train.marginal)
    return dynamics



torch.set_default_dtype(torch.float32)
parser = argparse.ArgumentParser(description='Learning the Random Language Model with deep neural networks')
parser.add_argument("--device", type=str, default='mps')
parser.add_argument("--datasets",type=str)



def default_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=default_device())
parser.add_argument('--dataset', type=str, default='rlm')
parser.add_argument('--mode', type=str, default='last')
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--seed_rules', type=int, default=12345678)    
parser.add_argument('--num_tokens', type=int, default=2)
parser.add_argument('--beta', type=float, default=2)
parser.add_argument('--train_size', type=int, default=2**15)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--test_size', type=int, default=2**15)
parser.add_argument('--seed_sample', type=int, default=56781234)
parser.add_argument('--input_format', type=str, default='long')
parser.add_argument('--whitening', type=int, default=0)       
parser.add_argument('--model', type=str, default='transformer_mlm')
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--ffwd_size', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--seed_model', type=int, default=12345678)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--scheduler', type=str, default='cosine-warmup')
parser.add_argument('--warmup_time', type=int, default=8)
parser.add_argument('--decay_time', type=int, default=2**18)
parser.add_argument('--max_epochs', type=int, default=16)
parser.add_argument('--print_freq', type=int, default=32768)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--measure_train', action='store_true', default=False)
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--checkpoints', default=False, action='store_true')


parser.add_argument('--outname', type=str, required=True, help='path of the output file')
config = parser.parse_args()
print(1)
dynamics =run(config)
