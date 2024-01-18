import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn

from src.model.Transformer import Transformer
from src.utils import source_vocabulary
from src.utils import translation_vocabulary
from src.utils import translation_to_index, index_to_translation ,source_to_index
from src.utils import START_TOKEN, END_TOKEN ,PADDING_TOKEN
from src.utils import create_masks
import os
import sys
from src.exception import CustomException
from src.logger import logging

source = 'artifacts/source.txt' 
translation = 'artifacts/translation.txt'


with open(source, 'r') as file:

    source_sentences = file.readlines()
    
with open(translation, 'r') as file:
    translation_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 200000
source_sentences = source_sentences[:TOTAL_SENTENCES]
translation_sentences = translation_sentences[:TOTAL_SENTENCES]
source_sentences = [sentence.rstrip('\n').lower() for sentence in source_sentences]
translation_sentences = [sentence.rstrip('\n').lower() for sentence in translation_sentences]

max_sequence_length = 200

def is_valid_tokens(sentence, vocab):
    return all(token in vocab for token in set(sentence))

def is_valid_length(sentence, max_length):
    return len(sentence) < max_length - 1

valid_sentence_indices = [
    index
    for index, (trans_sentence, source_sentence) in enumerate(zip(translation_sentences, source_sentences))
    if is_valid_length(trans_sentence, max_sequence_length) 
    and is_valid_length(source_sentence, max_sequence_length) 
    and is_valid_tokens(trans_sentence, translation_vocabulary)
]

translation_sentences = [translation_sentences[i] for i in valid_sentence_indices]
source_sentences = [source_sentences[i] for i in valid_sentence_indices]


print(f"Number of sentences: {len(translation_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indices)}")
import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 50
translation_vocab_size = len(translation_vocabulary)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          translation_vocab_size,
                          source_to_index,
                          translation_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)

print(Transformer)

class TextDataset(Dataset):

    def __init__(self, source_sentences, translation_sentences):
        self.source_sentences = source_sentences
        self.translation_sentences = translation_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.translation_sentences[idx]
    

dataset = TextDataset(source_sentences, translation_sentences)   

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)



criterian = nn.CrossEntropyLoss(ignore_index=translation_to_index[PADDING_TOKEN],
                                reduction='none')

for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        source_batch, translation_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(source_batch, translation_batch)
        optim.zero_grad()
        translation_predictions = transformer(source_batch,
                                     translation_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)

        labels = transformer.decoder.sentence_embedding.batch_tokenize(translation_batch, start_token=False, end_token=True)
        loss = criterian(
            translation_predictions.view(-1, translation_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == translation_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"source: {source_batch[0]}")
            print(f"Translation: {translation_batch[0]}")
            translation_sentence_predicted = torch.argmax(translation_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in translation_sentence_predicted:
              if idx == translation_to_index[END_TOKEN]:
                break
              predicted_sentence += index_to_translation[idx.item()]
            print(f"Prediction: {predicted_sentence}")


            transformer.eval()
            translation_sentence = ("",)
            source_sentence = ("saturday february 18 1995",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(source_sentence, translation_sentence)
                predictions = transformer(source_sentence,
                                          translation_sentence,
                                          encoder_self_attention_mask.to(device),
                                          decoder_self_attention_mask.to(device),
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] 
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_translation[next_token_index]
                translation_sentence = (translation_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break

            print(f"saturday february 18 1995 : {translation_sentence}")
            print("-------------------------------------------")



torch.save(transformer.state_dict(), 'artifacts/transformer_model.pth')