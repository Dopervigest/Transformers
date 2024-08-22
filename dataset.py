import torch
from torch import nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1 # EOS not used
        
        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError('Sentence is too long')
        
        # Add special tokens and pad
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64)
        ])
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64)
        ])
        
        # Target (what we want to get from decoder)
        label = torch.cat([
                        torch.tensor(dec_input_tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64)
                        ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return{
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_len) & (1, Seq_len, Seq_len)  
            "label" : label, # (seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text
             }

def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask == 0
    
    
class MonolingualDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len, truncation=True) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer = tokenizer        
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

        self.truncation = truncation
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        text = self.ds[index]['text']
        
        input_tokens = self.tokenizer.encode(text).ids
        num_padding = self.seq_len - len(input_tokens) - 1 # EOS not used

        if num_padding < 0:
            if self.truncation == True:
                input_tokens = input_tokens[:self.seq_len-1]
                num_padding = 0
            else:
                raise ValueError('Sentence is too long')
        
        gpt_input = torch.cat([
            self.sos_token,
            torch.tensor(input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * num_padding, dtype=torch.int64)
        ])
        
        # Target (what we want to get from decoder)
        label = torch.cat([
                        torch.tensor(input_tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * num_padding, dtype=torch.int64)
                        ])
        
        
        assert gpt_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return{
            "gpt_input": gpt_input, # (seq_len)
            "mask" : (gpt_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(gpt_input.size(0)), # (1, Seq_len) & (1, Seq_len, Seq_len)  
            "label" : label, # (seq_len)
            'text': text,
             }