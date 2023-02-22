import torch
from tokenizers import Tokenizer
from typing import NamedTuple

class IterativeSampler():
    '''Sampler that iteratively generates the next token in the sequence.'''''

    class IdsAndMask(NamedTuple):
        '''NamedTuple for storing the ids and mask of a batch of sequences.'''
        ids: torch.Tensor
        mask: torch.Tensor
    
    class IdsMaskAndLabels(NamedTuple):
        '''NamedTuple for storing the ids, mask and labels of a batch of sequences.'''
        ids: torch.Tensor
        mask: torch.Tensor
        labels: torch.Tensor

    def __init__(self, training_device: torch.device):
        self.training_device = training_device

    def sample(self, model: torch.nn.Module, batch: "dict[str,torch.Tensor]|IdsAndMask|IdsMaskAndLabels", tokenizer: Tokenizer):
        '''Sample from the model using the given batch as input. Takes arguments:
        model: The model to sample from.
        batch: The batch of sequences to use as input. Can be a dict of tensors or a NamedTuple.
        tokenizer: The tokenizer used to encode the sequences.'''


        mask_token = tokenizer.token_to_id('[MASK]')
        end_token = tokenizer.token_to_id('[END]')
        model.eval()
        with torch.no_grad():

            if isinstance(batch, dict):
                input_ids = batch['ids'].to(dtype=torch.long).to(self.training_device)
                attention_mask = batch['mask'].to(dtype=torch.bool).to(self.training_device)
            elif isinstance(batch, self.IdsAndMask):
                input_ids = batch.ids.to(dtype=torch.long).to(self.training_device)
                attention_mask = batch.mask.to(dtype=torch.bool).to(self.training_device)
            elif isinstance(batch, self.IdsMaskAndLabels):
                input_ids = batch.ids.to(dtype=torch.long).to(self.training_device)
                attention_mask = batch.mask.to(dtype=torch.bool).to(self.training_device)
            else:
                raise ValueError(f'Invalid batch type: {type(batch)}')

            indices = torch.tensor([range(input_ids.shape[1])]).to(self.training_device)
            expanded_indices = indices.expand(input_ids.shape) # Matrix of same shape as input_ids with the indices regarding the sequence dimension
            zeros = torch.zeros(input_ids.shape).to(self.training_device)
            next_tokens_mask = input_ids == mask_token # Get mask matching [MASK] tokens
            mask_indices = torch.where(next_tokens_mask, expanded_indices, zeros) # Get the indices of the mask tokens
            mask_indices = mask_indices.sum(dim=1) # Sum over the sequence dimension
            mask_indices_vector = mask_indices.unsqueeze(1) # Keep the vector form for updating the values
            mask_indices = mask_indices_vector.expand(input_ids.shape) # Expand to match input_ids shape

            finished_generation_vector = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(self.training_device)
            finished_generation_vector = finished_generation_vector.unsqueeze(1) # Keep the vector form for while loop
            finished_generation = finished_generation_vector.expand(input_ids.shape) # Expanded form for masking
            while not torch.all(finished_generation_vector):

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                next_tokens = torch.argmax(output.logits, dim=-1)
                input_ids[next_tokens_mask] = next_tokens[next_tokens_mask]
                
                generated_end_tokens = torch.where(next_tokens_mask, next_tokens, end_token)
                generated_end_tokens = torch.all(generated_end_tokens == end_token, dim=1) # Apply all over the sequence dimension
                finished_generation_vector = finished_generation_vector | generated_end_tokens # Update finished_generation_vector

                mask_indices_vector += 1 # Updates the mask_indices
                next_tokens_mask = mask_indices == expanded_indices
                next_tokens_mask = next_tokens_mask & ~finished_generation
                input_ids = torch.where(next_tokens_mask, mask_token, input_ids)
                attention_mask = attention_mask | next_tokens_mask

        return input_ids
