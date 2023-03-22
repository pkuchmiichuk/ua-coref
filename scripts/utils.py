import json
import jsonlines
from itertools import groupby
from operator import itemgetter


def find_slice_ids(sent_ids):
    sentence_lengths = [len(list(g)) for k, g in groupby(sent_ids)]
    slice_ids = []
    start = 0
    end = 0
    for length in sentence_lengths:
        end += length
        slice_ids.append((start, end))
        start = end

    return slice_ids


def split_into_sentences(text, sent_ids):
    slice_ids = find_slice_ids(sent_ids)
    sentences = [text[start:end] for start, end in slice_ids]
    return sentences


def flatten(l):
    return {item for sublist in l for item in sublist}


def flatten_double(l):
    # [[[1, 2], [5, 7]]] => [1, 2, 5, 7]
    return {item for top_list in l for sub_list in top_list for item in sub_list}


def mask(text, clusters):
    flat_clusters = flatten_double(clusters)
    id_mask = [i if i in flat_clusters else -1 for i, word in enumerate(text)]

    return id_mask


def find_ranges(data):
    ranges = []
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append([group[0], group[-1] + 1])
    return ranges


def read_data(filename):
    with jsonlines.open(filename, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    data = []
    for obj in lst:
        dct = {k: v for k, v in obj.items() if k in ['document_id', 'cased_words', 'sent_id', 'clusters']}
        data.append(dct)

    return data


def dump_jsonl(data, output_path, append=False):
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')
    print(f'Wrote to {output_path}')