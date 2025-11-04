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


def run( config):

    config.input_size = 2**config.num_layers
    #config.num_batches = config.train_size//config.batch_size
    #config.max_iters = config.max_epochs*config.num_batches
    config.train_size = (config.max_iters*config.batch_size)//config.max_epochs
    config.num_data = config.train_size + config.test_size

    print(config.device)

    print(f"Training for {config.max_iters} steps")
    
    rlm = datasets.RLM(
        v=config.num_features,
        L=config.num_layers,
        beta=config.beta,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,
        num_data=config.train_size+config.test_size,
        probs=None,
        transform=None
    )  
    inputs = rlm.trees[config.num_layers]
    targets = rlm.trees[0]
    train_loader, test_loader = init.init_data(inputs, targets, config)

    model = init.init_model(config)
    model0 = copy.deepcopy( model)

    criterion, optimizer, scheduler = init.init_training( model, config)
    print_ckpts, save_ckpts = init.init_loglinckpt( config.print_freq, config.max_iters, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    step=0
    dynamics, best = init.init_output(model, criterion, train_loader, test_loader, config)

    if config.checkpoints:
        torch.save(
            {'config': config, 'rules': rlm.M},
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

    for epoch in range(config.max_epochs):
    
        model.train()
        optimizer.zero_grad()
        running_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs.to(config.device))
            loss = criterion(outputs, targets.to(config.device))
            running_loss += loss.item()
            loss /= config.accumulation
            loss.backward()

            if ((batch_idx+1)%config.accumulation==0):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                if step==print_ckpt:
                    test_loss, test_acc = measures.test(model, criterion, test_loader, config.device)
                    if test_loss<best['loss']: # update best model if loss is smaller
                        best['step'] = step
                        best['loss'] = test_loss
                        best['model'] = copy.deepcopy( model.state_dict())

                    print('step : ',step, '\t running loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ', test loss: {:06.4f}'.format(test_loss))
                    print_ckpt = next(print_ckpts)

                    if step>=save_ckpt:

                        print(f'Checkpoint at step {step}, saving data ...')
                        save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                        if config.measure_train:
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
                                'entropy': rlm.entropy,
                                'dynamics': dynamics,
                                'step': step
                            }
                            torch.save(
                                {'config': config, 'output':output},
                                f"{config.outname}.pt"
                            )
                        save_ckpt = next(save_ckpts)
                            
                        
        if (running_loss/(batch_idx+1)) <= config.loss_threshold:

            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            if config.measure_train:
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
                    'entropy': rlm.entropy,
                    'dynamics': dynamics,
                    'step': step
                }
                torch.save(
                        output,
                        f"{config.outname}_t{step}.pt"
                    )
            break
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
