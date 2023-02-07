"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from tokenizers import Tokenizer
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--iterative_building", action="store_true")
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False,
        load_for_iterative_building=args.iterative_building
    )

    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')
    
    from tqdm import tqdm

    if args.step == args.diffusion_steps:
        args.use_ddim = False
        step_gap = 1
    else:
        args.use_ddim = True
        step_gap = args.diffusion_steps//args.step

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    for cond in tqdm(all_test_data):
        padding_token_id = tokenizer.pad_token_id
        sep_token_id = tokenizer.sep_token_id
        end_token_id = tokenizer.tokenizer.token_to_id('[END]') if isinstance(tokenizer.tokenizer, Tokenizer) else tokenizer.sep_token_id

        finished_iterative_building = False
        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        reference_ids = input_ids_x.clone().detach()
        input_ids_mask = cond.pop('input_mask').to(dtype=th.bool ,device=dist_util.dev()) # [batch, seq_len]
        input_ids_mask_ori = input_ids_mask.clone().detach()
        generated_end_tokens_or_reached_max_len = th.zeros(input_ids_x.shape[0], dtype=th.bool, device=dist_util.dev()) # [batch]
        generated_end_tokens_or_reached_max_len = generated_end_tokens_or_reached_max_len.unsqueeze(1)  # [batch, 1]
        generated_end_tokens_or_reached_max_len_mask = generated_end_tokens_or_reached_max_len.expand(input_ids_mask.shape) # [batch, seq_len]
        padding_column = th.full_like(input_ids_x[:, :1], padding_token_id, device=dist_util.dev())
        while True:
            input_ids_x_clone = input_ids_x.clone().detach()
            input_ids_mask_clone = input_ids_mask.clone().detach()
            ### input_ids_x: [batch_size, seq_len]
            ### is structured as [x_1, ..., x_n, [END], y_1, ..., y_ji, [SEP], [PAD], ..., [PAD]] in iteration i
            x_start = model.get_embeds(input_ids_x_clone)

            noise = th.randn_like(x_start)
            input_ids_mask_clone = th.broadcast_to(input_ids_mask_clone.unsqueeze(dim=-1), x_start.shape)
            x_noised = th.where(input_ids_mask_clone==0, x_start, noise)

            model_kwargs = {}

            sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

            samples_history : th.Tensor = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask_clone,
                x_start=x_start,
                gap=step_gap
            )

            if args.iterative_building:
                ### samples: [batch_size, seq_len, output_dim]
                ### structured as [x_1, ..., x_n, [END], y_1, ..., y_ji, [SEP], y_(ji+1), ..., y_j(i+1), [PAD], ..., [PAD]] in each row in iteration i (counting from 0)
                ### do knn for y_(ji+1), ..., y_j(i+1) and update input_ids_x
                samples = samples_history[-1].clone().detach()

                distances_to_embeddings : th.Tensor = model.get_logits(samples)  # bsz, seqlen, vocab

                nearest_embeddings = distances_to_embeddings.argmax(dim=-1) # bsz, seqlen
                
                generated_tokens_mask = input_ids_mask & ( nearest_embeddings != padding_token_id )

                generated_end_tokens = input_ids_mask & ( nearest_embeddings == end_token_id )
                sequence_generated_end_token = ~th.all(~generated_end_tokens, dim=1, keepdim=True) # bsz, 1
                reached_max_len = nearest_embeddings[:, -1:] != padding_token_id # bsz, 1
                stopped_generating = th.all(~generated_tokens_mask, dim=1, keepdim=True) # bsz, 1

                generated_end_tokens_or_reached_max_len = generated_end_tokens_or_reached_max_len | sequence_generated_end_token | reached_max_len | stopped_generating	
                generated_end_tokens_or_reached_max_len_mask = generated_end_tokens_or_reached_max_len.expand(input_ids_mask.shape)

                # Shift the nearest embeddings to the left by one position and pad the last position with padding_token_id
                nearest_embeddings = th.cat(
                    [nearest_embeddings[:, 1:], padding_column], dim=1
                )

                shifted_generated_tokens_mask = th.cat(
                    [generated_tokens_mask[:, 1:], th.zeros_like(generated_tokens_mask[:, :1], dtype=th.bool, device=dist_util.dev())], dim=1
                )

                set_tokens_mask = th.logical_or(input_ids_mask, input_ids_x == sep_token_id)

                set_sep_token_mask =  generated_tokens_mask & ~shifted_generated_tokens_mask

                nearest_embeddings = th.where(set_sep_token_mask, sep_token_id, nearest_embeddings)

                input_ids_x = th.where(set_tokens_mask, nearest_embeddings, input_ids_x)

                input_ids_mask = input_ids_mask & ~generated_tokens_mask & ~generated_end_tokens_or_reached_max_len_mask

                finished_iterative_building = th.all(generated_end_tokens_or_reached_max_len)


            if not args.iterative_building:
                break
            elif finished_iterative_building:
                samples = [model.get_embeds(input_ids_x)]
                break

        

        model_emb_copy.cpu()
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []


        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()
        # print('decoding for seq2seq', )
        # print(arr.shape)

        reshaped_x_t = x_t
        distances_to_embeddings = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        cands = th.topk(distances_to_embeddings, k=1, dim=-1)
        sample = cands.indices
        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(reference_ids, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
        fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')

if __name__ == "__main__":
    main()
