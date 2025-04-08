#!/usr/bin/env python3

# Mostly adapted from parts of
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py and
# https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/indexed_dataset.py

import sys
import struct

import numpy as np

from argparse import ArgumentParser

from transformers import AutoTokenizer


_INDEX_HEADER = b"MMIDIDX\x00\x00"


DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('path', help='data path without suffix (.idx/.bin)')
    ap.add_argument('tokenizer', nargs='?')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--stop', type=int, default=None)
    ap.add_argument('--idx-only', action='store_true')
    ap.add_argument('--separate-tokens', action='store_true')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    idxfn, binfn = f'{args.path}.idx', f'{args.path}.bin'

    # Load index
    print(f'reading .idx {idxfn}', flush=True)
    with open(idxfn, 'rb') as f:
        header = f.read(len(_INDEX_HEADER))
        assert header == _INDEX_HEADER
        version = struct.unpack('<Q', f.read(8))
        assert version == (1,)

        code = struct.unpack('<B', f.read(1))[0]
        dtype = DTYPE_MAP[code]
        print(f'dtype         : {dtype.__name__}', flush=True)

        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
        print(f'sequence_count: {sequence_count}', flush=True)
        print(f'document_count: {document_count}', flush=True)

        sequence_lengths = np.fromfile(f, np.int32, sequence_count)
        print(f'sequence_lengths (shape {sequence_lengths.shape}): {sequence_lengths}', flush=True)

        sequence_pointers = np.fromfile(f, np.int64, sequence_count)
        print(f'sequence_pointers (shape {sequence_pointers.shape}): {sequence_pointers}', flush=True)

        document_indices = np.fromfile(f, np.int64, document_count)
        print(f'document_indices (shape {document_indices.shape}): {document_indices}', flush=True)

        total_tokens = int(sequence_pointers[-1]/dtype().itemsize + sequence_lengths[-1])
        print(f'total tokens {total_tokens:}', flush=True)

    # Sanity check vs. https://github.com/NVIDIA/Megatron-LM/issues/1519
    assert not np.any(sequence_lengths < 0), 'negative value(s) in sequence_lengths'
    assert not np.any(sequence_pointers < 0), 'negative value(s) in sequence_pointers'
    assert not np.any(document_count < 0), 'negative value(s) in document_count'

    if args.idx_only:
        return 0

    # Load and print decoded .bin records
    print(f'reading .bin {binfn}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(binfn, 'rb') as f:
        stop = args.stop if args.stop is not None else sequence_count
        for i in range(args.start, stop, args.step):
            offset, length = sequence_pointers[i], sequence_lengths[i]
            f.seek(offset)
            data = np.fromfile(f, dtype, length)
            print('-'* 30, i, '-'*30)
            if args.separate_tokens:
                print(tokenizer.convert_ids_to_tokens(data))
            else:
                print(tokenizer.decode(data))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
