import argparse
import itertools
from projection import *
from utils import read_data, dump_jsonl
from transformers import AutoTokenizer, AutoModel, pipeline


def align_and_project(text, clusters, sent_ids, translator_pipeline):
    entity_map = create_entity_map(text, clusters, sent_ids)
    new_target_ids = create_target_sent_ids(entity_map, translator_pipeline, model, tokenizer)
    result = [(source_sent, text_ent, source_ent, target_sent, target_ent) for
              ((source_sent, text_ent, source_ent), (target_sent, target_ent)) in zip(entity_map, new_target_ids)]
    final_map = create_final_entity_mapping(result)
    translations = [x[3] for x in result]
    target_sent_ids = [list(itertools.repeat(i, len(x))) for i, x in enumerate(translations)]
    # flatten
    target_sent_ids = [item for sublst in target_sent_ids for item in sublst]
    target_text = [word for sentence in translations for word in sentence]
    target_clusters = create_target_clusters(clusters, target_text, final_map)

    return target_text, target_clusters, target_sent_ids


def process_split(split, output, translator, verbose):
    data = read_data(split)
    print(f"\nWorking on {split}:")

    for i, document in enumerate(tqdm.auto.tqdm(data)):
        s_document_id = document['document_id']
        s_text = document['cased_words']
        s_sent_id = document['sent_id']
        s_clusters = document['clusters']

        if verbose:
            print(f"\nWorking on document {i}:")
        t_text, t_clusters, t_sent_id = align_and_project(s_text, s_clusters, s_sent_id, translator)

        target_dict = dict()

        target_dict['document_id'] = s_document_id
        target_dict['cased_words'] = t_text
        target_dict['sent_id'] = t_sent_id
        target_dict['clusters'] = t_clusters

        dump_jsonl(target_dict, output, append=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silver data creation")
    parser.add_argument("--translation_model",
                        help="pretrained translation model checkpoint",
                        default="Helsinki-NLP/opus-mt-en-uk", type=str)
    parser.add_argument("--input_dir",
                        help="directory containing English OntoNotes data",
                        default="ontonotes-ua", type=str)
    parser.add_argument("--output_dir",
                        help="output directory",
                        default="ontonotes-ua", type=str)
    parser.add_argument("-train", "--process_train",
                        help="whether to process the train set, False by default",
                        action="store_true")
    parser.add_argument("-dev", "--process_dev",
                        help="whether to process the dev set, False by default",
                        action="store_true")
    parser.add_argument("-test", "--process_test",
                        help="whether to process the test set, False by default",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="increase verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.process_train == args.process_dev == args.process_test == False:
        print("Error: Please specify at least one data split to process.")
    else:
        checkpoint = args.translation_model
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint, output_attentions=True)
        translator = pipeline("translation", model=checkpoint)

        TRAIN = args.input_dir + '/english_train.jsonlines'
        DEV = args.input_dir + '/english_development.jsonlines'
        TEST = args.input_dir + '/english_test.jsonlines'

        OUTPUT_TRAIN = args.output_dir + '/uk_silver_train.jsonlines'
        OUTPUT_DEV = args.output_dir + '/uk_silver_development.jsonlines'
        OUTPUT_TEST = args.output_dir + '/uk_silver_test.jsonlines'

        for flag, split, output in zip((args.process_train, args.process_dev, args.process_test),
                                       (TRAIN, DEV, TEST),
                                       (OUTPUT_TRAIN, OUTPUT_DEV, OUTPUT_TEST)):
            if flag:
                process_split(split, output, translator, args.verbose)
