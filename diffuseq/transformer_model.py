from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
from fairseq.models.transformer import TransformerEncoder as FairseqEncoder
from fairseq.models.transformer import TransformerDecoder as FairseqDecoder
from fairseq.models.transformer import TransformerModel as FairseqTransformerModel
from fairseq.models.transformer.transformer_config import TransformerConfig
from .fairseq_config import load_fairseq_config
from argparse import Namespace
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class BertModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding. Based on the BertModel from transformers.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
    
class FairseqEncoderModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding. Based on Fairseq implementation.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dictionary,
        dropout=0,
        config={},
        config_name='transformer_iwslt_de_en',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        config_args = Namespace(**{
            "no_token_positional_embeddings": True,
            "layernorm_embedding": False,
            "encoder_normalize_before": False,
            "dropout": dropout,
            **config
        })

        load_fairseq_config(config_name=config_name, args=config_args)

        transformer_config = TransformerConfig.from_namespace(config_args)

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = transformer_config.encoder.embed_dim

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        if self.input_dims != self.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, self.hidden_size),
                                              nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
        
        assert init_pretrained == 'no', 'init_pretrained is not supported for FairseqModel'
        # FairseqEncoder loads padding_idx, so we need to set it after loading the embedding
        self.word_embedding.padding_idx = dictionary.pad()
        self.input_transformers = FairseqEncoder(transformer_config, dictionary, embed_tokens=self.word_embedding)

        self.register_buffer("position_ids", torch.arange(transformer_config.max_source_positions).expand((1, -1)))
        self.position_embeddings = nn.Embedding(transformer_config.max_source_positions, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        
        # FairseqEncoder applies dropout itself
        # self.dropout = nn.Dropout(transformer_config.hidden_dropout_prob)

        self.lm_head = nn.Sequential(
            nn.Linear(self.hidden_size, vocab_size),
            nn.Softmax(dim=-1)
        )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return F.linear(hidden_repr, self.word_embedding.weight)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.word_embedding.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.word_embedding.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0)
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.LayerNorm(emb_inputs)

        # Fairseq transformer needs the tokens to calculate a padding mask
        with torch.no_grad():
            guessed_tokens = self.get_logits(x).argmax(-1) # [batch_size, seq_len]
            # Assure not all tokens are padding
            # if (guessed_tokens == self.word_embedding.padding_idx).all():
            #     guessed_tokens.fill_(self.word_embedding.padding_idx + 1)

        input_trans_hidden_states = self.input_transformers(guessed_tokens, token_embeddings=emb_inputs)['encoder_out'][0] # [seq_len, batch_size, hidden_size]
        input_trans_hidden_states = input_trans_hidden_states.transpose(0, 1) # [batch_size, seq_len, hidden_size]
        
        probability_vectors = self.lm_head(input_trans_hidden_states) # [batch_size, seq_len, vocab_size]
        mean = F.linear(probability_vectors, self.word_embedding.weight.t()) # [batch_size, seq_len, input_dims]
        mean = mean.type(x.dtype)
        return mean
    

class FairseqEncoderDecoderModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding. Based on Fairseq implementation.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dictionary,
        dropout=0,
        config={},
        config_name='transformer_iwslt_de_en',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=2,
    ):
        super().__init__()

        config_args = Namespace(**{
            "no_token_positional_embeddings": True,
            "layernorm_embedding": False,
            "encoder_normalize_before": False,
            "dropout": dropout,
            **config
        })

        load_fairseq_config(config_name=config_name, args=config_args)

        transformer_config = TransformerConfig.from_namespace(config_args)

        assert input_dims == transformer_config.encoder.embed_dim == transformer_config.decoder.embed_dim, \
            "input_dims should be equal to transformer_config.encoder.embed_dim and transformer_config.decoder.embed_dim"

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = transformer_config.encoder.embed_dim

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        if self.input_dims != self.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, self.hidden_size),
                                              nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
        
        assert init_pretrained == 'no', 'init_pretrained is not supported for FairseqModel'
        # FairseqEncoder loads padding_idx, so we need to set it after loading the embedding
        self.word_embedding.padding_idx = dictionary.pad()
        self.encoder = FairseqEncoder(transformer_config, dictionary, embed_tokens=self.word_embedding)
        self.decoder = FairseqDecoder(transformer_config, dictionary, embed_tokens=self.word_embedding, no_encoder_attn=transformer_config.no_cross_attention)

        self.register_buffer("position_ids", torch.arange(transformer_config.max_source_positions).expand((1, -1)))
        self.position_embeddings = nn.Embedding(transformer_config.max_source_positions, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        
        # FairseqEncoder applies dropout itself
        # self.dropout = nn.Dropout(transformer_config.hidden_dropout_prob)

        if not transformer_config.adaptive_softmax_cutoff:
            self.lm_head = nn.Softmax(dim=-1)
            

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return F.linear(hidden_repr, self.word_embedding.weight)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.word_embedding.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.word_embedding.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.LayerNorm(emb_inputs)

        # Fairseq transformer needs the tokens to calculate a padding mask
        with torch.no_grad():
            guessed_tokens = self.get_logits(x).argmax(-1) # [batch_size, seq_len]
            # Assure not all tokens are padding
            # if (guessed_tokens == self.word_embedding.padding_idx).all():
            #     guessed_tokens.fill_(self.word_embedding.padding_idx + 1)

        encoder_out = self.encoder(guessed_tokens, token_embeddings=emb_inputs) # [seq_len, batch_size, hidden_size]
        decoder_out, _ = self.decoder(guessed_tokens, encoder_out=encoder_out) # [batch_size, seq_len, vocab_size]
        
        probability_vectors = self.lm_head(decoder_out) if self.lm_head is not None else decoder_out # [batch_size, seq_len, vocab_size]
        mean = F.linear(probability_vectors, self.word_embedding.weight.t()) # [batch_size, seq_len, input_dims]
        mean = mean.type(x.dtype)
        return mean