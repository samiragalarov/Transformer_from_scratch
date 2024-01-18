import os
import sys
import torch
from src.model.Transformer import Transformer 
import numpy as np
from src.utils import source_vocabulary
from src.utils import translation_vocabulary
from src.utils import translation_to_index, index_to_translation
from src.utils import START_TOKEN, END_TOKEN ,PADDING_TOKEN
from src.utils import create_masks




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


source_to_index = {v:k for k,v in enumerate(source_vocabulary)}


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






model_path = 'artifacts/transformer_model.pth'
transformer.load_state_dict(torch.load(model_path, map_location=device))


transformer.to(device)


transformer.eval()


def generate_translation(max_length, source_sentence):

    translation_sentence = ("",)
    for word_counter in range(max_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(source_sentence, translation_sentence)
        predictions = transformer.to(device)(
            source_sentence,
            translation_sentence,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False
        )
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_translation[next_token_index]
        translation_sentence = (translation_sentence[0] + next_token, )
        if next_token == END_TOKEN:
            break
    return translation_sentence


# print(generate_translation(max_sequence_length, source_sentence=("saturday fruary 13 1995",)))
