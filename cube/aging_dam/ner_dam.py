import spacy
from collections import defaultdict
from tqdm import tqdm
import json

from IPython import embed

# Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

watershed_word = ['streams', 'rivers', 'brooks', 'creeks', 'washes', 'lakes', 'stream', 'river', 'brook', 'creek', 'wash', 'lake']

# read corpus into date_list
date_list = []
with open('corpus/aging_dam/corpus.txt') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        date_list.append(line.strip())

print(f'Length of date_list: {len(date_list)}')



# # ================ Process whole documents and extract for one ENTITY ================
# with open('hypercube/aging_dam/person.txt','w') as fout: 
#     for t in tqdm(date_list):
#         text = (t)
#         doc = nlp(text)
#         entity_dict = defaultdict(int)

#         # Find named entities, phrases and concepts
#         for entity in doc.ents:
#             # if entity.label_ in ['LOC', 'GPE', 'FAC', 'ORG' 'PERSON', "DATE", "EVENT"]:
#             if entity.label_ in ['PERSON']:  
#                 entity_dict[entity.text] += 1     # entity.label_: LOC, entity.text: Lake Dunlap
#                 # text_seq = entity.text.split()
#                 # if text_seq[0].lower() in watershed_word or text_seq[-1].lower() in watershed_word:
#                 #     entity_dict[entity.text] += 1
#         fout.write(json.dumps(entity_dict) + '\n')


# ================ Process whole documents and extract for one ENTITY ================
with open('hypercube/aging_dam/quant.txt','w') as fout:
    for t in tqdm(date_list):
        text = (t)
        doc = nlp(text)
        entity_dict = defaultdict(int)
        print("entity_dict:", entity_dict)

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            if entity.label_ in ['QUANTITY']:  # ['ORG' 'PERSON', "DATE", "EVENT", 'QUANTITY']
                entity_dict[entity.text] += 1
        fout.write(json.dumps(entity_dict) + '\n')


# # ================ Process whole documents and extract for MERGED ENTITY ================
# ENTITY_MAPPING = {
#     "GPE": "LOCATION",  # Merge GPE into LOCATION
#     "LOC": "LOCATION",  # Merge LOC into LOCATION
# }

# # Process documents and extract entities
# with open('hypercube/aging_dam/location.txt', 'w') as fout:
#     for t in tqdm(date_list):
#         doc = nlp(t)
#         entity_dict = defaultdict(int)

#         # Merge GPE and EVENT into a single category
#         for entity in doc.ents:
#             entity_label = ENTITY_MAPPING.get(entity.label_)
#             if entity_label:
#                 # entity_dict[entity_label + ": " + entity.text] += 1
#                 entity_dict[entity.text] += 1

#         fout.write(json.dumps(entity_dict) + '\n')



