import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .rnn_decoder import AttnDecoderRNN
from typing import Callable, NamedTuple
from itertools import islice
from diffuseq.text_datasets import infinite_loader
import json
import os
import random
import tqdm
import gc

DEFAULTS = {
    'MAX_LENGTH': 10,
    'TEACHER_FORCING_RATIO': 0.5,
}

if os.path.exists('rnn_decoder/defaults.json'):
    with open('rnn_decoder/defaults.json', 'r') as f:
        DEFAULTS = json.load(f)

MAX_LENGTH = DEFAULTS['MAX_LENGTH']
TEACHER_FORCING_RATIO = DEFAULTS['TEACHER_FORCING_RATIO']

class Evaluator(NamedTuple):
    '''Evaluator for the RNNDecoderTrainer. This NamedTuple contains the following fields:
    name: The name of the evaluator
    evaluation_function: The function to call to evaluate the model. This function takes a BertEncoderOnlyTrainer as input and returns a float.'''
    metrics: "list[str]"
    evaluation_function: Callable[["dict[str, torch.Tensor]"], "dict[str, float]"]

class RNNDecoderTrainer():
    '''Based on the training script from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html'''

    def __init__(self, train_dataloader: DataLoader, sos_token: int, eos_token: int, pad_token: int, decoder: AttnDecoderRNN, optimizer: optim.Optimizer, criterion, device: torch.device):
        self.train_dataloader = train_dataloader
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.do_eval = False
        self.do_checkpoint = False
        self.logger = lambda _: None

    def set_logger(self, logger: Callable[[str], None]):
        self.logger = logger

    def enable_eval(self, eval_dataloader: DataLoader, eval_interval: int, eval_batches = 1):
        self.eval_dataloader = infinite_loader(eval_dataloader)
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.do_eval = True

        def default_eval(eval_batch):

            input_tensor = eval_batch['reference'].to(device=self.device, dtype=torch.long)
            target_tensor = eval_batch['target'].to(device=self.device, dtype=torch.long)
            encoded_tensor = eval_batch['encoded'].to(device=self.device, dtype=torch.float)

            batch_size = input_tensor.shape[0]

            encoder_outputs = encoded_tensor.view(batch_size, 1, -1) # (batch_size, 1, seq_len*input_dim)
            decoder_hidden = self.decoder.initHidden(batch_size, self.device, encoder_outputs=encoder_outputs) # (1, batch_size, hidden_size)

            decoder_input = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=self.device) 

            loss = 0 

            # Without teacher forcing: use its own predictions as the next input
            generated_eos_tokens = torch.full((batch_size, 1),  False, dtype=torch.bool, device=self.device)
            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoded_tensor)
                topi = decoder_output.argmax(dim=2) # (batch_size, 1)
                decoder_input = topi.detach()  # detach from history as input

                decoder_output = decoder_output.squeeze(1)
                loss += self.criterion(decoder_output, target_tensor[:,di])

                generated_eos_tokens = generated_eos_tokens | (decoder_input == self.eos_token)
                if all(generated_eos_tokens):
                    break

            loss = loss / MAX_LENGTH

            return {
                'eval_loss': loss
            }

        self.eval_callbacks = [
            Evaluator(
                metrics=['eval_loss'],
                evaluation_function=default_eval
            )
        ]

    def register_eval_callback(self, callback: Evaluator):
        assert self.eval_dataloader is not None, 'You must call enable_eval before calling register_eval_callback'
        self.eval_callbacks.append(callback)

    def enable_checkpointing(self, checkpoint_callback: Callable[["dict"], None], checkpoint_interval: int):
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_interval = checkpoint_interval
        self.do_checkpoint = True

    def train(self, epochs: int, use_tqdm = True, start_epoch = 0):

        self.decoder.train()
        self.decoder.to(self.device)
        
        for epoch in range(epochs):
            if use_tqdm:
                loop = tqdm.tqdm(self.train_dataloader, desc=f'Epoch {epoch + start_epoch}')
            else:
                loop = self.train_dataloader
        
            for n, batch in enumerate(loop):
                input_tensor = batch['reference'].to(device=self.device, dtype=torch.long)
                target_tensor = batch['target'].to(device=self.device, dtype=torch.long)
                encoded_tensor = batch['encoded'].to(device=self.device, dtype=torch.float)

                loss = self.step(input_tensor, target_tensor, encoded_tensor)
                if use_tqdm:
                    loop.set_postfix(loss=loss)
                
                if self.do_eval and n % self.eval_interval == 0:
                    eval_results = self.eval()
                    self.logger(eval_results)
                    if use_tqdm:
                        loop.set_postfix(eval_results)

                if self.do_checkpoint and n % self.checkpoint_interval == 0:
                    params = self.decoder.state_dict()
                    self.checkpoint_callback(params)

    def eval(self):
        self.decoder.eval()
        with torch.no_grad():
            
            results = {metric : .0 for callback in self.eval_callbacks for metric in callback.metrics}

            for batch in islice(self.eval_dataloader, self.eval_batches):
                for callback in self.eval_callbacks:
                    eval_results = callback.evaluation_function(batch)
                    for metric in callback.metrics:
                        results[metric] += eval_results[metric]
            for metric in results:
                results[metric] /= self.eval_batches

        self.decoder.train()
        return results

    def step(self, input_tensor, target_tensor, encoded_tensor):
        self.optimizer.zero_grad()

        batch_size = input_tensor.size(0)
        longest_target_length = torch.max(torch.sum(target_tensor != self.pad_token, dim=1)).item()

        loss = 0

        decoder_input = torch.full((batch_size,1), self.sos_token, dtype=torch.long, device=self.device) # (batch_size, 1)

        encoder_outputs = encoded_tensor.view(batch_size, 1, -1) # (batch_size, 1, seq_len*input_dim)
        decoder_hidden = self.decoder.initHidden(batch_size, self.device, encoder_outputs=encoder_outputs) # (1, batch_size, hidden_size)

        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(longest_target_length):
                decoder_output, decoder_hidden, _decoder_attention = self.decoder(decoder_input, decoder_hidden, encoded_tensor)
                decoder_output = decoder_output.squeeze(1)
                loss += self.criterion(decoder_output, target_tensor[:,di])
                decoder_input = target_tensor[:,di:di+1]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(longest_target_length):
                decoder_output, decoder_hidden, _decoder_attention = self.decoder(decoder_input, decoder_hidden, encoded_tensor)
                topi = decoder_output.argmax(dim=2) # (batch_size, 1)
                topi = topi.to(dtype=torch.long)
                decoder_input = topi.detach()  # detach from history as input (batch_size, 1)
                decoder_output = decoder_output.squeeze(1)
                loss += self.criterion(decoder_output, target_tensor[:,di])
                if torch.all((decoder_input == self.eos_token) | (decoder_input == self.pad_token)):
                    break

        loss.backward()

        self.optimizer.step()

        return loss.item() / longest_target_length