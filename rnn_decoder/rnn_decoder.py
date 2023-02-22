### Here we want to try using an RNN Decoder instead of simply applying knn
import torch
from torch import nn
from torch.nn import functional as F
import json
import os

DEFAULTS = {
    'HIDDEN_SIZE': 256,
    'OUTPUT_SIZE': 256,
    'DROPOUT_P': 0.1,
    'MAX_LENGTH': 10,
    'NUM_LAYERS': 1
}

if os.path.exists('rnn_decoder/defaults.json'):
    with open('rnn_decoder/defaults.json', 'r') as f:
        DEFAULTS = json.load(f)

HIDDEN_SIZE = DEFAULTS['HIDDEN_SIZE']
OUTPUT_SIZE = DEFAULTS['OUTPUT_SIZE']
DROPOUT_P = DEFAULTS['DROPOUT_P']
MAX_LENGTH = DEFAULTS['MAX_LENGTH']
NUM_LAYERS = DEFAULTS['NUM_LAYERS']

class AttnDecoderRNN(nn.Module):
    '''Implementation from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html'''

    def __init__(self, encoder_hidden_size, encoder_embedding_dimension, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE, dropout_p=DROPOUT_P, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.hidden_state_encoder = nn.Linear(encoder_hidden_size, self.num_layers * self.hidden_size)
        if encoder_embedding_dimension != self.hidden_size:
            self.embedding_upscaling = nn.Linear(encoder_embedding_dimension, self.hidden_size)
            self.apply_embedding_upscaling = True
        else:
            self.apply_embedding_upscaling = False

        if self.num_layers > 1:
            self.reduce_hidden_state = nn.Linear(self.num_layers * self.hidden_size, self.hidden_size)
            self.apply_reduce_hidden_state = True
        else:
            self.apply_reduce_hidden_state = False

    def forward(self, input_tokens, hidden, encoder_outputs):
        embedded = self.embedding(input_tokens) # (batch_size, 1, hidden_size)
        embedded = self.dropout(embedded) # (batch_size, 1, hidden_size)

        if self.apply_reduce_hidden_state: # If we have more than one layer, we need to reduce the hidden state
            reduced_hidden = hidden.view(-1, self.num_layers * self.hidden_size) # (batch_size, num_layers * hidden_size)
            reduced_hidden = self.reduce_hidden_state(reduced_hidden) # (batch_size, hidden_size)
            reduced_hidden = reduced_hidden.view(-1, 1, self.hidden_size) # (batch_size, 1, hidden_size)
        else:
            reduced_hidden = hidden.transpose(0, 1) # (batch_size, 1, hidden_size)
        embedded_and_hidden = torch.cat((embedded, reduced_hidden), 2) # (batch_size, 1, hidden_size * 2)

        attentions = self.attn(embedded_and_hidden) # (batch_size, 1, max_length)
        attentions = F.softmax(attentions, dim=2) # (batch_size, 1, max_length)

        # Encoder outputs are (batch_size, seq_len, input_dim) | seq_len = max_length
        encoder_outputs_upscaled = self.embedding_upscaling(encoder_outputs) if self.apply_embedding_upscaling else encoder_outputs # (batch_size, max_length, hidden_size)
        attn_applied = torch.bmm(attentions, encoder_outputs_upscaled) # (batch_size, 1, hidden_size)

        output = torch.cat((embedded, attn_applied), 2) # (batch_size, 1, hidden_size * 2)
        output = self.attn_combine(output) # (batch_size, 1, hidden_size)

        output = F.relu(output) # (batch_size, 1, hidden_size)
        output, hidden = self.gru(output, hidden) # (batch_size, 1, hidden_size), (num_layers, batch_size, hidden_size)

        output = self.out(output) # (batch_size, 1, output_size)
        output = F.log_softmax(output, dim=2) # (batch_size, 1, output_size)
        return output, hidden, attentions

    def initHidden(self, batch_size, device, encoder_outputs=None):
        '''Initialize the hidden state of the decoder. If encoder_outputs are provided, use a linear layer to initialize the hidden state.'''
        if encoder_outputs is not None:
            encoder_outputs.to(device)
            hidden = self.hidden_state_encoder(encoder_outputs)
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
            return hidden
        else:
            return torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)