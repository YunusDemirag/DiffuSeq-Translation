# Tool to undo BPE subword tokenization

import sys
import argparse
import json

parser = argparse.ArgumentParser(description='Undo BPE subword tokenization')
parser.add_argument('--file', '-f', type=str, help='File formatted as a jsonl file with texts having been tokenized with BPE')
parser.add_argument('--bpe', '-b', default='|', required=False, type=str, help='BPE subword character')

args = parser.parse_args()

with open(args.file, 'r') as f:
    for line in f:
        line = json.loads(line)
        reconstruced_line = {}
        # Remove BPE subword character for every field in the jsonl file
        # And join the tokens back together
        for key in line:
            reconstruced = ''
            # Split the tokens by whitespace
            for token in line[key].split(' '):
                if token.startswith(args.bpe):
                    reconstruced += token[1:]
                elif token == '.':
                    reconstruced += token
                else:
                    reconstruced += ' ' + token
            reconstruced_line[key] = reconstruced
        print(json.dumps(reconstruced_line))
            
