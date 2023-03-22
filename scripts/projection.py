import tqdm
import string
import tokenize_uk
from utils import *
from alignment import *


def create_entity_map(text, clusters, sent_ids):
    masked = mask(text, clusters)

    entities = [x for cluster in clusters for x in cluster]

    sentences = split_into_sentences(text, sent_ids)
    masks = split_into_sentences(masked, sent_ids)

    masked_sentences = list(zip(sentences, masks))

    ent_map = []
    num_words = 0

    for sentence, msk in masked_sentences:
        text_entities = []
        sentence_entities = []
        # all items equal == -1, no entities in sentence
        if len(set(msk)) == 1 and msk[0] == -1:
            text_entities.append([])
            sentence_entities.append([])
        else:
            # there are entities in sentence
            if -1 in msk:
                flat_ent_ids = [x for x in msk if x != -1]
            else:
                flat_ent_ids = [x for x in msk]
            for ent in entities:
                if ent[0] in flat_ent_ids:
                    start, end = ent
                    text_entity = [x for x in range(start, end)]
                    c_start = start - num_words
                    c_end = end - num_words
                    sentence_entity = [x for x in range(c_start, c_end)]

                    text_entities.append(text_entity)
                    sentence_entities.append(sentence_entity)

        num_words += len(sentence)
        # hardcoded bugfix
        if text_entities == []:
            text_entities = [[]]
            sentence_entities = [[]]
        ent_map.append((text_entities, sentence_entities))

    return [(sent, text_ent, sent_ent) for sent, (text_ent, sent_ent) in zip(sentences, ent_map)]


def create_target_sent_ids(source_entity_mapping, translator_pipeline, model, tokenizer):
    translations = []
    target_sent_ids_mapping = []

    for sentence in tqdm.auto.tqdm(source_entity_mapping):
        snt_tokenized, text_ids_mapping, sent_ids_mapping = sentence

        snt_as_string = " ".join(snt_tokenized)
        translation = translator_pipeline(snt_as_string)[0]['translation_text']

        # tokenizing using a Ukrainian tokenization library (better than splitting by whitespaces)
        translation_tokenized = tokenize_uk.tokenize_words(translation)

        # saving correctly tokenized sentences here already
        # but since we align based on an MT model, and it has its own tokenization, there could be mismatches in tokens
        # hence updating translations later based on MT tokens is another option
        translations.append(translation_tokenized)

        # nonetheless, combining correct tokens with whitespaces later as alignment works better this way
        translation_whitespaced = " ".join(translation_tokenized)

        target_sent_ids = []

        if text_ids_mapping[0]:
            # if there is a coreferring entity in the sentence, align based on cross-attention
            alignment = align_base(snt_as_string, translation_whitespaced, model, tokenizer)
            # extract the source words that were aligned (aligning based on decoder (target sentence) first, so some words of the source might not appear)
            aligned_source_words = {pair[1][0] for pair in alignment}

            # for every span (a list of ALL word indices in the span) that denotes an entity
            for ids_container in sent_ids_mapping:
                target_ids_container = []
                # for counter, word position in the sentence
                for i, word_id in enumerate(ids_container):
                    # find the word in the sentence by position
                    word = snt_tokenized[word_id]
                    # if this word was matched with some target word AND it is this specific word (its index is in the container we're looking at)
                    if word in aligned_source_words and word_id in ids_container:
                        # word was aligned with something
                        aligned_with = []
                        # for every alignment pair of structure ((target_token, token_index), (source_token, token_index)) -- as one source word could match to many target words
                        for pair in alignment:
                            # if the specific source word was aligned
                            if pair[1] == (word, word_id):
                                # save the target word id
                                aligned_with.append(pair[0][1])
                        # save all the target word ids that match with a specific source word
                        target_ids_container.append(aligned_with)
                    else:
                        # word was not aligned, add empty structure
                        target_ids_container.append([])
                # save all the alignments for a specific entity
                target_sent_ids.append(target_ids_container)
            # save all the alignments for all entities in a sentence
            target_sent_ids_mapping.append(target_sent_ids)
        else:
            # there is no coreferring entity in the sentence, append empty structure
            target_sent_ids_mapping.append([[[]]])

    return list(zip(translations, target_sent_ids_mapping))


def create_final_entity_mapping(prev_map):
    final_entity_mapping = dict()
    num_target_words = 0

    for s in prev_map:
        source_sent, text_ent, source_ent, target_sent, target_ent = s
        # pair the entities together
        for s_ent, t_ent in zip(text_ent, target_ent):
            if s_ent:
                clean_t_ent = find_ranges([x + num_target_words for x in sorted(flatten(t_ent))])
                final_entity_mapping[s_ent[0], s_ent[-1] + 1] = clean_t_ent
        num_target_words += len(target_sent)

    return final_entity_mapping


def create_target_clusters(source_clusters, target_text, mapping):
    raw_target_clusters = []
    clean_target_clusters = []
    for cluster in source_clusters:
        current_target_cluster = []
        for entity in cluster:
            target_entities = mapping[tuple(entity)]
            for entity in target_entities:
                current_target_cluster.append(entity)
        if current_target_cluster:
            raw_target_clusters.append(current_target_cluster)

    for cluster in raw_target_clusters:
        current_cluster = []
        for t_ent in cluster:
            span = target_text[t_ent[0]:t_ent[1]]  # list of strings

            # this removes punctuation as a separate cluster
            if len(span) == 1 and span[0] in string.punctuation:
                continue

            # this should strip punctuation from well-formed entities; it is super clunky, hence not including (TODO: rewrite)

            # elif span[0] in string.punctuation:
            #   t_ent[0] += 1
            #   if span[-1] in string.punctuation:
            #     t_ent[-1] -= 1
            #     current_cluster.append(t_ent)
            #   else:
            #     current_cluster.append(t_ent)
            # elif span[-1] in string.punctuation:
            #   t_ent[-1] -= 1
            #   current_cluster.append(t_ent)

            else:
                current_cluster.append(t_ent)
        clean_target_clusters.append(current_cluster)

    return clean_target_clusters