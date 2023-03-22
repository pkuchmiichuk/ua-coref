import torch
import re


def num_layers(attention):
    return len(attention)


def num_heads(attention):
    return attention[0][0].size(0)


def extract_attention(source_sentence, target_sentence, model, tokenizer):
    encoder_input_ids = tokenizer(source_sentence, return_tensors="pt", add_special_tokens=True).input_ids
    with tokenizer.as_target_tokenizer():
        decoder_input_ids = tokenizer(target_sentence, return_tensors="pt", add_special_tokens=True).input_ids

    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

    encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
    decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

    cross_attention = outputs.cross_attentions

    # legacy from bertviz implementation, probably not needed, but didn't delete just in case for now
    include_layers = list(range(num_layers(cross_attention)))
    include_heads = list(range(num_heads(cross_attention)))

    attention = [cross_attention[layer_index] for layer_index in include_layers]

    squeezed = []
    for layer_attention in attention:
        layer_attention = layer_attention.squeeze(0)
        layer_attention = layer_attention[include_heads]
        squeezed.append(layer_attention)

    stacked = torch.stack(squeezed)

    correct_head = stacked[1][0]
    corresponding_tokens = torch.topk(correct_head, 2, dim=1)[1].tolist()

    # if max is attention to the </s> token, take second to last as max
    clean_attention = []
    for item in corresponding_tokens:
        if item[0] == len(encoder_text) - 1:
            clean_attention.append(item[1])
        else:
            clean_attention.append(item[0])
    return encoder_text, decoder_text, clean_attention


def prettify_tokens(tokens):
    token_dict = dict()
    for i, subtoken in enumerate(tokens):
        # beginning of the token, start new token
        if subtoken[0] == "▁" or subtoken == '</s>':
            # append memorized token from previous iterations
            if i != 0:
                token_dict[token + "***" + str(i)] = token_ids
            token = subtoken[1:]
            token_ids = [i]
        # inside of the token, update previous token
        else:
            token += subtoken
            token_ids.append(i)
    # append end of sentence token
    token_dict['</s>'] = [len(tokens) - 1]

    for i, k in enumerate(token_dict):
        token_dict[k] = (i, token_dict[k])

    return token_dict


def _align(source_tokens, target_tokens, alignment):
    aligned = []
    for i, pointer in enumerate(alignment):
        for k, v in target_tokens.items():
            if i in v[1]:
                tgt_token = re.sub(r'\*{3}.+', "", k)
                tgt_token_id = v[0]
                break
        for k, v in source_tokens.items():
            if pointer in v[1]:
                src_token = re.sub(r'\*{3}.+', "", k)
                src_token_id = v[0]
                break
        pair = ((tgt_token, tgt_token_id), (src_token, src_token_id))
        if pair not in aligned:
            aligned.append(pair)
    return aligned


def align_base(source_sentence, target_sentence, model, tokenizer):
    source_text, target_text, attention = extract_attention(source_sentence, target_sentence, model, tokenizer)

    source_tokens = prettify_tokens(source_text)
    target_tokens = prettify_tokens(target_text)

    aligned = _align(source_tokens, target_tokens, attention)

    return aligned[:-1]