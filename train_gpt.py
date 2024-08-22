import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import MonolingualDataset, causal_mask
from gpt_model import build_GPT
from config import get_weights_file_path, get_gpt_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from tqdm import tqdm
import warnings

from pathlib import Path

def get_all_sentences(ds):
    for item in ds:
        yield item['text']

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'].format(config['category']))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens= ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('McAuley-Lab/Amazon-Reviews-2023', config['category'], trust_remote_code=True)['full']
    
    # Build tokenizers
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    # keep 90% for training and 10% for val
    train_ds_size = int(config['train_size'] * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = MonolingualDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = MonolingualDataset(val_ds_raw, tokenizer, config['seq_len'])
    
    max_len = 0
    
    for item in ds_raw:
        ids = tokenizer.encode(item['text']).ids
        max_len = max(max_len, len(ids))
    
    print(f'Max len of sentence: {max_len}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_len):
    model = build_GPT(vocab_size=vocab_len, seq_len=config['seq_len'], d_model = config['d_model'], N = config['num_blocks'])    
    return model


def greedy_decode(model, input_tokens, mask, tokenizer, max_len, device):
        sos_idx = tokenizer.token_to_id('[SOS]')
        eos_idx = tokenizer.token_to_id('[EOS]')
        
        decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(input_tokens).to(device)
        while True:
            if decoder_input.size(1) >= max_len:
                break
            
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(mask).to(device)
            out = model(decoder_input, decoder_mask)
            
            prob = model.project(out[:, -1])
            
            # token with max probability
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(input_tokens).fill_(next_word.item()).to(device)], dim=1)
            
            if next_word == eos_idx:
                break
        return decoder_input.squeeze()
    

def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer=None, num_examples=2):
    model.eval()
    count = 0

    predicted = []
    
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            input_tokens = batch['gpt_input'].to(device)
            mask = batch['mask'].to(device)
            
            assert input_tokens.size(0) == 1, 'Batch size must be 1 for validation'
            
            model_out = greedy_decode(model, input_tokens, mask, tokenizer, max_len, device)
            
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            predicted.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f'PREDICTION: {model_out_text}')
            
            if count >= num_examples:
                break


def train_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size())
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f' Preloading model {model_filename}')
        
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    model.to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            gpt_input = batch['gpt_input'].to(device) # (B, seq_len)
            mask = batch['mask'].to(device) # (B, 1, seq_len, seq_len)
            
            output = model(gpt_input, mask) # (B, Seq_len, d_model)
            proj_output = model.project(output) # (B, Seq_len, tgt_vocab_size)
            
            label = batch['label'].to(device) # (B, seq_len)
            
            # (B, Seq_len, tgt_vocab_size) --> # (B * Seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({'Loss': f'{loss.item():6.3f}'})
            
            writer.add_scalar('train loss', loss.item(), global_step)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, 
                                lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)
            
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_gpt_config()
    train_model(config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    