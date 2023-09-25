########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import time

import numpy as np
import torch
from torch.nn import functional as F

class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
        self.alpha_decay = alpha_decay # gradually decay the penalty
        self.token_ban = token_ban # ban the generation of some tokens
        self.token_stop = token_stop # stop generation whenever you see any token here
        self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def encode_num(self, x):
        list = []
        tokens = []
        for i in x:
            token, length = self.encode(i)
            list.append(length)
            tokens += token
        return tokens, list
    def encode_batch(self, x):
        batch = []
        for i in x:
            token = self.tokenizer.encode(i)
            batch.append(token)
        tokens = torch.tensor(batch)
        return tokens
    def decode_bsz(self, x):
        list = []
        for i in x:
            i = [int(i)]
            t = self.tokenizer.decode(i)
            list.append(t)
        return np.array(list, dtype='U')

    def decode_batch(self, x):
        return self.tokenizer.decode_batch(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.5, top_k=0):
        
        logits = logits.squeeze(0)
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    
    #cfg 用于多轮对话，更专注于整段对话逻辑
    def cfg_logits(self, logits,  logits1, aph=1,temperature=1.0, top_p=0.5, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        probs1 = F.softmax(logits1.float(), dim=-1)
        #probs = aph*probs-(aph-1)*probs1
        probs = aph*(probs-probs1)+probs1
        top_k = int(top_k)
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        print('out', out.device)
        return int(out)

    def sample_bsz(self, logits, temperature=1.0, top_p=0.5, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = sorted_probs[torch.arange(probs.shape[0]),np.argmax(cumulative_probs > top_p,1)]
        probs[probs < cutoff.unsqueeze(1)] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[top_k:]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[:,0]
        out = out.unsqueeze(1).cpu()
        return out


    #动态整理队列 先生成的答案先出
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        B = len(ctx)
        all_str = {}
        all_state = {}
        set_n = np.arange(B)
        out_np = np.empty((B,), dtype='U')
        for i in range(token_count):
            # forward & adjust prob.
            tokens, lengs = self.encode_num(ctx) if i == 0 else (token, [0])
            out, state = self.model.forward(tokens, state, lengs)
            torch.cuda.synchronize()

            token = self.sample_bsz(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            # output

            tmp = self.decode_bsz(token)
            k = len(tmp)-1
            while k >= 0:
                if '\n\n' in tmp[k] or '\ufffd' in tmp[k] or '\n\n' in out_np[k]:
                    all_str[set_n[k]] = out_np[k]
                    state_list = []
                    for t, s in enumerate(state):
                        state_list.append(s[k])
                        if k == len(tmp) - 1:
                            state[t] = state[t][:-1, :]
                        else:
                            state[t] = torch.cat((state[t][:k, :], state[t][k + 1:, :]), dim=0)
                    all_state[set_n[k]] = state_list
                    set_n = np.delete(set_n, k, axis=0)
                    out_np = np.delete(out_np, k, axis=0)
                    token = np.delete(token, k, axis=0)
                    tmp = np.delete(tmp, k, axis=0)
                if len(set_n) == 0:
                    return all_str, all_state
                k -= 1
            out_np = np.char.add(out_np, tmp)
        for k in range(len(tmp)):
            all_str[set_n[k]] = out_np[k]
            all_state[set_n[k]] = state_list

        return all_str, all_state

