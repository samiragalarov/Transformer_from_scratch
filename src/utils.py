import sys
import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn



START_TOKEN = ''
PADDING_TOKEN = ''
END_TOKEN = ''

source_vocabulary= [START_TOKEN,' ', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y',
                         PADDING_TOKEN, END_TOKEN]

translation_vocabulary = [START_TOKEN, ' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', PADDING_TOKEN, END_TOKEN]


d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 50
translation_vocab_size = len(translation_vocabulary)

index_to_translation = {k:v for k,v in enumerate(translation_vocabulary)}
translation_to_index = {v:k for k,v in enumerate(translation_vocabulary)}
index_to_source = {k:v for k,v in enumerate(source_vocabulary)}
source_to_index = {v:k for k,v in enumerate(source_vocabulary)}



NEG_INFTY = -1e9

def create_masks(source_batch,translation_batch):
    num_sentences = len(source_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      source_sentence_length, translation_sentence_length = len(source_batch[idx]), len(translation_batch[idx])
      source_chars_to_padding_mask = np.arange(source_sentence_length + 1, max_sequence_length)
      translation_chars_to_padding_mask = np.arange(translation_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, source_chars_to_padding_mask] = True
      encoder_padding_mask[idx, source_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, translation_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, translation_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, source_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, translation_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask
