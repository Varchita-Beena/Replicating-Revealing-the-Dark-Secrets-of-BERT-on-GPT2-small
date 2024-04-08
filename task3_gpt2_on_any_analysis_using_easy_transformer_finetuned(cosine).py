#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F
from easy_transformer import EasyTransformer
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.functional import cosine_similarity


# In[2]:


valid_dataset = load_dataset('glue', 'mrpc', split='validation')


# In[3]:


valid_dataset[0], len(valid_dataset)


# In[4]:


num_samples = 100
subset_indices = torch.randperm(len(valid_dataset)).tolist()[:num_samples]
valid_dataset = valid_dataset.select(subset_indices)


# In[5]:


len(valid_dataset)


# In[6]:


valid_dataset[0]


# In[7]:


def tokenize(datapoint, max_length = 1024, token_to_add = 50256):
    sep_place = [0]
    sentence1 = datapoint['sentence1']
    sentence1_tokens = reference_gpt2.to_tokens(sentence1)
    
    sep_2 = sentence1_tokens.size(1)
    sep_place.append(sep_2+1)
    sentence2 = datapoint['sentence1']
    sentence2_tokens = reference_gpt2.to_tokens(sentence2)
    concatenated_tokens = torch.cat((sentence1_tokens, sentence2_tokens), dim=1)
    
    labels = torch.tensor(datapoint['label'])
    real_length = concatenated_tokens.size(1)
    remaining_length = max_length - concatenated_tokens.size(1)
    while remaining_length > 0:
        concatenated_tokens = torch.cat((concatenated_tokens, torch.tensor([[token_to_add]])), dim=1)
        remaining_length -= 1
    return concatenated_tokens, labels, real_length, sep_place


# In[8]:


class CustomGPT2ForSequenceClassification(EasyTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.unembed = None
        self.classification_head1 = torch.nn.Linear(config.d_model * config.n_ctx, num_labels)        
        
    def forward(self, input_ids):
       
        embed = self.embed(tokens=input_ids)
        embed = embed.squeeze(1)
        pos_embed = self.pos_embed(input_ids)
        residual = embed + pos_embed
        for block in self.blocks:
            normalized_resid_pre = block.ln1(residual)
            attn_out = block.attn(normalized_resid_pre)
            resid_mid = residual + attn_out
            normalized_resid_mid = block.ln2(resid_mid)
            mlp_out = block.mlp(normalized_resid_mid)
            resid_post = resid_mid + mlp_out
        normalized_resid_final = self.ln_final(resid_post)
        normalized_resid_final = normalized_resid_final.view(normalized_resid_final.shape[0], -1)
        logits = self.classification_head1(normalized_resid_final)
        return logits
    


# In[9]:


def register_hooks(module):
    def hook(module, input, output):
        print("Output shape:", output.shape)  
    module.register_forward_hook(hook)


# In[10]:


def register_attention_hooks(module):
    if isinstance(module, EasyTransformer):
        for i, block in enumerate(module.blocks):
            attention_module = block.attn.hook_attn
            def hook(module, input, output):
                attention_scores = output[0]
                attention_scores_list.append(attention_scores)
            attention_module.register_forward_hook(hook)
            


# In[11]:


def register_attention_hooks_ref(module):
    if isinstance(module, EasyTransformer):
        for i, block in enumerate(module.blocks):
            attention_module = block.attn.hook_attn
            def hook(module, input, output):
                attention_scores = output[0]
                attention_scores_list_ref.append(attention_scores)
            attention_module.register_forward_hook(hook)
            


# In[12]:


def get_cosine(pre, fine):
    cosines = []
    for i, every in enumerate(pre):
        attention_scores1_flat = pre[i].view(-1)
        attention_scores2_flat = fine[i].view(-1)
        print()
        cosine_sim = cosine_similarity(attention_scores1_flat.unsqueeze(0), attention_scores2_flat.unsqueeze(0), dim=1)
        cosines.append(cosine_sim.item())
    return cosines


# In[13]:


num_labels = 2
model_path = '../trained_models/easy_transformer_gpt2small_mrpc.pth' 

cosines = []  
for i, point in enumerate(valid_dataset): 
    temp = []
    print('data point ',i)
    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    config = reference_gpt2.cfg
    model = CustomGPT2ForSequenceClassification(config)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("mps")
    
    model.to(device)
    model.eval()
    reference_gpt2.to(device)
    reference_gpt2.eval()
    
    tokens, label, length, seperators = tokenize(point)
    
    attention_scores_list = []
    register_attention_hooks(model)
    outputs = model(tokens)
    
    attention_scores_list_ref = []
    register_attention_hooks_ref(reference_gpt2)
    outputs_ref = reference_gpt2(tokens)
    
    for i in range(0, 12):
        cosine_values = get_cosine(attention_scores_list[i], attention_scores_list_ref[i])
        temp.append(cosine_values)
    cosines.append(temp)
        


# In[14]:


mean_array = np.mean(np.array(cosines), axis = 0)


# In[15]:


plt.imshow(mean_array, cmap='Blues_r')
plt.colorbar()
plt.title('Mean Array')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.savefig('../gpt2_small/imgs/mrpc_cosines.png')
plt.show()


# In[16]:


np.save('../gpt2_small/arrays/cosines_mrpc.npy', cosines)


# In[ ]:




