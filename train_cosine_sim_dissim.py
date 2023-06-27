import os
import time
import random
from copy import deepcopy

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
import numpy as np
from tqdm import tqdm
from bpemb import BPEmb
from icecream import ic
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs

from tri_model import TriModel
from trimodal_dataset import CosineSimDataset, collate_trimodal_cosine
from objectives import cosine_sim_dissim_objective


def train_epoch(train_dataloader, dev_dataloader, model, optimizer, scheduler, 
    criterion, accelerator, save_dict, logger):
    
    model.train()
    train_info = save_dict['train_info']
    curr_epoch = train_info['curr_epoch']
    step_loss = 0.0
    total_loss = 0.0
    n_steps = len(train_dataloader)
    start_time = time.time()
    optimizer.zero_grad()
    
    for i, sample in tqdm(enumerate(train_dataloader), 
        total=len(train_dataloader), desc=f'Training, epoch {curr_epoch}', 
        disable = not accelerator.is_local_main_process):
        
        with accelerator.autocast():
            embeds = model(sample['text_batch'], sample['audio_batch'], sample['video_batch'])
            loss = criterion(embeds)
        
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0, norm_type=2)

        if (i+1) % train_info['accumulate_train_batches'] == 0 or (i+1) == n_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_loss += loss.item()
        total_loss += loss.item()

        if accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]['lr']
            if (i+1) % train_info['log_steps'] == 0:
                cur_loss = step_loss / train_info['log_steps']
                ms_per_batch = (time.time() - start_time) * 1000 / train_info['log_steps']
                
                train_info['curr_step'] = i+1
                train_info['avg_step_train_losses'].append(cur_loss)
                train_info['avg_ms_per_batch'].append(ms_per_batch)

                accelerator.print(f'| epoch {curr_epoch:3d} | step {i+1:5d} / {n_steps:5d} batches ' +
                    f'| milli-sec/batch {ms_per_batch:7.2f} | loss {cur_loss:7.2f} |')
                logger.log({'lr': lr, 'train/step/#': (i+1), 'train/step/loss': cur_loss, 
                    'train/step/ms_per_batch': ms_per_batch})

                step_loss = 0.0
                start_time = time.time()

        if (i+1) % train_info['validate_every_n_steps'] == 0:
            accelerator.print('Starting dev evaluation')
            
            dev_start = time.time()
            dev_loss, metric_vals = evaluate(dev_dataloader, model, criterion, accelerator)
            if type(dev_loss) == torch.Tensor:
                dev_loss = accelerator.gather(dev_loss).mean().cpu().item()
            accelerator.wait_for_everyone()
            model.train()
            
            if accelerator.is_local_main_process:
                eval_duration = time.time() - dev_start
                train_info['dev_losses'].append(dev_loss)
                train_info['dev_metrics'].append(metric_vals)
                train_info['dev_durations'].append(eval_duration)

                accelerator.print(f'|| epoch {curr_epoch:3d} | ' +
                f'| dev eval duration {eval_duration:7.2f} sec | dev loss {dev_loss:5.2f} ||')
                accelerator.print(f'Dev metrics:', metric_vals)
                logger.log({'dev/duration': eval_duration, 'dev/loss': dev_loss})
                for metric in metric_vals:
                    logger.log({f'dev/{metric}': metric_vals[metric]})
                
                if metric_vals['audio_video_dist'] < train_info['best_audio_video_dist']:
                    train_info['best_audio_video_dist'] = metric_vals['audio_video_dist']
                    save_checkpoint('best.pth', model, optimizer, accelerator, save_dict, 
                        store_optimizer_state=False)
                    accelerator.print('*'*10, 'Updated as best checkpoint', '*'*10)
    lr = optimizer.param_groups[0]['lr']
    
    return total_loss / n_steps, lr


@torch.no_grad()
def evaluate(eval_dataloader, model, criterion, accelerator):
    model.eval()
    losses = 0.0
    t_a_ds = list()
    t_v_ds = list()
    a_v_ds = list()

    for sample in tqdm(eval_dataloader, total=len(eval_dataloader), 
        desc='Evaluating', disable = not accelerator.is_local_main_process):
        
        embeds = model(sample['text_batch'], sample['audio_batch'], sample['video_batch'])
        loss = criterion(embeds)
        loss_mean = accelerator.gather(loss).mean()
        losses += loss_mean.item()
        
        te = embeds['text_emb']
        ae = embeds['audio_emb']
        ve = embeds['video_emb']
        t_a_d = (1 - nn.functional.cosine_similarity(te, ae, dim=-1)).tolist()
        t_v_d = (1 - nn.functional.cosine_similarity(te, ve, dim=-1)).tolist()
        a_v_d = (1 - nn.functional.cosine_similarity(ae, ve, dim=-1)).tolist()
        t_a_ds.extend(t_a_d)
        t_v_ds.extend(t_v_d)
        a_v_ds.extend(a_v_d)
    loss = losses / len(eval_dataloader)
    metrics = {
        'text_audio_dist': np.array(t_a_ds).mean(),
        'text_video_dist': np.array(t_v_ds).mean(),
        'audio_video_dist': np.array(a_v_ds).mean()
    }

    return loss, metrics


def save_checkpoint(filename, model, optimizer, accelerator, save_dict, store_optimizer_state=False):
    os.makedirs(save_dict['experiment_name'], exist_ok=True)
    
    save_dict = deepcopy(save_dict)
    unwrapped_model = accelerator.unwrap_model(model)
    save_dict['model_state_dict'] = unwrapped_model.state_dict()
    if store_optimizer_state:
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        save_dict['optimizer_state_dict'] = unwrapped_optimizer.state_dict()
    accelerator.save(save_dict, os.path.join(save_dict['experiment_name'], filename))
    
    return


def train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, accelerator, save_dict, 
    logger):
    
    train_info = save_dict['train_info']
    for epoch in tqdm(range(1, train_info['total_epochs']+1), desc='Epochs', 
        disable = not accelerator.is_local_main_process):
        
        epoch_start_time = time.time()
        train_info['curr_epoch'] = epoch
        
        train_loss, lr = train_epoch(train_dataloader, dev_dataloader, model, optimizer, 
                                     scheduler, criterion, accelerator, save_dict, logger)

        accelerator.wait_for_everyone()
        accelerator.print(f'\nTraining complete for epoch {epoch} with average loss: {train_loss}, ' + 
            f'current learning rate {lr}')
        
        if accelerator.is_local_main_process:
            epoch_duration = time.time() - epoch_start_time
            train_info['epoch_durations'].append(epoch_duration)
            save_checkpoint(f'epoch-{epoch}.pth', model, optimizer, accelerator, save_dict, 
                store_optimizer_state=False)
        
            accelerator.print('-'*90)
            accelerator.print(f'| end of epoch {epoch:3d} | time: {epoch_duration:7.2f}s | ' + 
                f'train loss {train_loss:5.2f} |')
            accelerator.print('-'*90)
            logger.log({'epoch/#': epoch, 'epoch/duration': epoch_duration, 'epoch/loss': train_loss})

    return


if __name__ == '__main__':
    os.chdir('/scratch/sagarsj42')
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42'
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    EMB_SIZE = 300
    BPE_VOCAB_SIZE = 10000
    DATASET_INFO_DIR = './yt8m-clips-dataset-info'
    AUDIO_FEATURES_DIR = './yt8m-audio-features'
    VIDEO_FEATURES_DIR = './yt8m-video-features'
    AUDIO_MODEL_KEY = 'HTSAT-base'
    AUDIO_MODEL_PATH = 'music_audioset_epoch_15_esc_90.14.pt'
    VIDEO_MODEL_KEY = 'MCG-NJU/videomae-base'
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2 * TRAIN_BATCH_SIZE
    ACCUMULATE_TRAIN_BATCHES = 4
    N_EPOCHS = 5
    LEARNING_RATE = 6e-5
    LOG_STEPS = 50
    VALIDATE_PER_EPOCH = 0.5
    SEED = 15
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    EXP_NAME = 'cosine-sim-dissim'
    accelerator.print('Experiment name:', EXP_NAME)

    if not os.path.exists(AUDIO_MODEL_PATH):
        os.system(f'wget https://huggingface.co/lukewys/laion_clap/resolve/main/{AUDIO_MODEL_PATH}')
    
    text_bpe_model = BPEmb(lang='en', vs=BPE_VOCAB_SIZE, dim=EMB_SIZE)

    train_ds = CosineSimDataset('train', DATASET_INFO_DIR, text_bpe_model, 
                        AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    dev_ds = CosineSimDataset('dev', DATASET_INFO_DIR, text_bpe_model, 
                        AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    test_ds = CosineSimDataset('test', DATASET_INFO_DIR, text_bpe_model, 
                        AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    
    # from torch.utils.data import Subset
    # train_ds = Subset(train_ds, list(range(500)))
    # dev_ds = Subset(dev_ds, list(range(100)))
    # test_ds = Subset(test_ds, list(range(200)))
    
    if accelerator.is_local_main_process:
        ic(len(train_ds), len(dev_ds), len(test_ds))
    
    train_dl = DataLoader(train_ds, collate_fn=collate_trimodal_cosine, 
                        batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dev_dl = DataLoader(dev_ds, collate_fn=collate_trimodal_cosine, 
                        batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, collate_fn=collate_trimodal_cosine, 
                        batch_size=EVAL_BATCH_SIZE, shuffle=False)
    train_dl, dev_dl, test_dl = accelerator.prepare(train_dl, dev_dl, test_dl)
    
    if accelerator.is_local_main_process:
        ic(len(train_dl), len(dev_dl), len(test_dl))
    
    MODEL_ARGS = {
        'emb_size': EMB_SIZE,
        'bpe_vocab_size': BPE_VOCAB_SIZE,
        'audio_model_key': AUDIO_MODEL_KEY,
        'audio_model_path': AUDIO_MODEL_PATH,
        'video_model_key': VIDEO_MODEL_KEY
    }
    model = TriModel(**MODEL_ARGS)
    criterion = cosine_sim_dissim_objective
    
    OPTIMIZER_ARGS = {
        'lr': LEARNING_RATE,
        'betas': (0.9, 0.98),
        'weight_decay': 1e-2,
        'eps': 1e-9
    }
    optimizer = AdamW(model.parameters(), **OPTIMIZER_ARGS)
    
    num_train_update_steps = N_EPOCHS * len(train_dl) // ACCUMULATE_TRAIN_BATCHES
    SCHEDULER_ARGS = {
        'num_warmup_steps': num_train_update_steps // 10,
        'num_training_steps': num_train_update_steps
    }
    scheduler = get_linear_schedule_with_warmup(optimizer, **SCHEDULER_ARGS)

    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_local_main_process:
        ic(type(model), optimizer, scheduler)
    
    DATASET_INFO = {
        'train_dataset_size': len(train_ds),
        'dev_dataset_size': len(dev_ds),
        'test_dataset_size': len(test_ds)
    }
    
    TRAIN_INFO = {
        'per_device_train_batch_size': TRAIN_BATCH_SIZE,
        'per_device_eval_batch_size': EVAL_BATCH_SIZE,
        'accumulate_train_batches': ACCUMULATE_TRAIN_BATCHES,
        'total_epochs': N_EPOCHS,
        'per_device_train_steps': len(train_dl),
        'per_device_dev_steps': len(dev_dl),
        'log_steps': LOG_STEPS,
        'validate_per_epoch': VALIDATE_PER_EPOCH,
        'validate_every_n_steps': int(len(train_dl) * VALIDATE_PER_EPOCH),
        'curr_epoch': 0,
        'curr_step': 0,
        'best_audio_video_dist': float('inf'),
        'avg_step_train_losses': list(),
        'dev_losses': list(),
        'dev_metrics': list(),
        'dev_durations': list(),
        'avg_ms_per_batch': list(),
        'epoch_durations': list()
    }

    COMMENTS = ''
    
    SAVE_DICT = {
        'experiment_name': EXP_NAME,
        'model_args': MODEL_ARGS,
        'model_state_dict': {},
        'optimizer_args': OPTIMIZER_ARGS,
        'scheduler_args': SCHEDULER_ARGS,
        'dataset_info': DATASET_INFO,
        'train_info': TRAIN_INFO,
        'comments': COMMENTS
    }

    accelerator.print('State:', SAVE_DICT)
    
    if accelerator.is_local_main_process:
        logger = wandb.init(project=EXP_NAME, config=SAVE_DICT)
        train(train_dl, dev_dl, model, criterion, optimizer, scheduler, accelerator, SAVE_DICT, logger)
    else:
        train(train_dl, dev_dl, model, criterion, optimizer, scheduler, accelerator, SAVE_DICT, None)
    
    accelerator.print('Training complete.')
    accelerator.print('Evaluating on the test set')
    
    del model
    best_ckpt = torch.load(os.path.join(EXP_NAME, 'best.pth'))
    best_model = TriModel(**best_ckpt['model_args'])
    best_model.load_state_dict(best_ckpt['model_state_dict'])
    best_model = accelerator.prepare(best_model)
    test_loss, test_metrics = evaluate(test_dl, best_model, criterion, accelerator)
    
    accelerator.print('Test set loss:', test_loss)
    accelerator.print('Test set metrics:', test_metrics)
    accelerator.print('Evaluation complete.')
