import spacy
from collections import defaultdict
from tqdm import tqdm
import json
import re
from collections import Counter
from IPython import embed

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_trf")  


# read date_source
date_list = []
with open('corpus/geography/SciDCC-Geography.txt') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        date_list.append(line.strip())
print(f'Length of date_list: {len(date_list)}')


# # Process whole documents and extract for one ENTITY
# with open('geography/event.txt','w') as fout:
#     for t in tqdm(date_list):
#         text = (t)
#         doc = nlp(text)
#         entity_dict = defaultdict(int)
        
#         # Find named entities, phrases and concepts
#         for entity in doc.ents:
#             #print(f"entity.label_: {entity.label_}, entity.text: {entity.text}")

#             if entity.label_ in ['EVENT']:  # ['ORG' 'PERSON', "DATE", "EVENT"]
#                 entity_dict[entity.text] += 1
#         fout.write(json.dumps(entity_dict) + '\n')


# # Process whole documents and extract for MERGED ENTITY
# ENTITY_MAPPING = {
#     "GPE": "LOCATION",  # Merge GPE into LOCATION
#     "LOC": "LOCATION",  # Merge LOC into LOCATION
# }

# # Process documents and extract entities
# with open('geography/location.txt', 'w') as fout:
#     for t in tqdm(date_list):
#         doc = nlp(t)
#         entity_dict = defaultdict(int)

#         # Merge GPE and EVENT into a single category
#         for entity in doc.ents:
#             entity_label = ENTITY_MAPPING.get(entity.label_)
#             if entity_label:
#                 # entity_dict[entity_label + ": " + entity.text] += 1
#                 entity_dict[entity.text] += 1

#         fout.write(json.dumps(entity_dict) + '\n')  # Save results




# ================ theme ================
earthquake_pattern = re.compile(r'\bearthquake(s)?\b', re.IGNORECASE)  
gate_pattern = re.compile(r'\bgate(s)?\b', re.IGNORECASE)
mag_pattern = re.compile(r'\bmagnitude(s)?\b', re.IGNORECASE)
bird_pattern = re.compile(r'\bbird(s)?\b', re.IGNORECASE)
seal_pattern = re.compile(r'\bseal(s)?\b', re.IGNORECASE)
stopover_passage_pattern = re.compile(r'\stopover-to-passage(s)?\b', re.IGNORECASE)
crater_pattern = re.compile(r'\bcrater(s)?\b', re.IGNORECASE)
flood_pattern = re.compile(r'\bflood(s|ing)?\b', re.IGNORECASE)

# Process the text and count occurrences
with open('geography/theme.txt', 'w') as fout:
    for text in tqdm(date_list):
        theme_dict = Counter()
        
        # Capture continuous two words, e.g., "earthquake earthquake"
        words = text.lower().split()
        for i, word in enumerate(words):
            if words[i] == "earthquake" and words[i + 1] == "earthquake":
                # phrase = " ".join(words[max(0, i - 2): min(len(words), i + 3)])  # Capture 2 words before and after
                phrase = " ".join(words[i: min(len(words), i + 2)]) 
                theme_dict[phrase] += 1
            elif words[i] == "earthquake" and words[i + 1] == "magnitude":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "climate" and words[i + 1] == "change":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "bird" and words[i + 1] == "species":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "mantle" and words[i + 1] == "temperature":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "water" and words[i + 1] == "storage": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "tiger" and words[i + 1] == "shark": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "mountain" and words[i + 1] == "water":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "warming" and words[i + 1] == "trend": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "wind" and words[i + 1] == "turbine": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "recycled" and words[i + 1] == "crust": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "economic" and words[i + 1] == "output": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "carbon" and words[i + 1] == "dioxide": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "forest" and words[i + 1] == "loss": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "blue" and words[i + 1] == "starfish": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "food" and words[i + 1] == "web": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "Earth's" and words[i + 1] == "mantle": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "birds" and words[i + 1] == "breed":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "sponges" and words[i + 1] == "move":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "bluefin" and words[i + 1] == "tuna":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "global" and words[i + 1] == "precipitation":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "atlantic" and words[i + 1] == "sturgeon":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "wolf" and words[i + 1] == "spiders":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "aquaculture" and words[i + 1] == "sites":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "fisheries" and words[i + 1] == "management":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "carbon" and words[i + 1] == "sequestration":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1               
            elif words[i] == "early" and words[i + 1] == "climate": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "early" and words[i + 1] == "ocean": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "sea" and words[i + 1] == "currents": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "ice" and words[i + 1] == "melt": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "oil" and words[i + 1] == "palm": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "nutrient" and words[i + 1] == "supply": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "permafrost" and words[i + 1] == "thawing": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "soil" and words[i + 1] == "moisture": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "vize" and words[i + 1] == "island": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "physical" and words[i + 1] == "characteristics": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "ancient" and words[i + 1] == "dna": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "island" and words[i + 1] == "bird": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "meltwater" and words[i + 1] == "distribution": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "groundwater" and words[i + 1] == "recharge":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "mangrove" and words[i + 1] == "forests":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "global" and words[i + 1] == "mangrove":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
                 
            elif words[i] == "cutthroat" and words[i + 1] == "eels":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "plastic" and words[i + 1] == "debris":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "gene" and words[i + 1] == "expression":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "soil" and words[i + 1] == "moisture":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1     
            elif words[i] == "gas" and words[i + 1] == "leakage":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "amur" and words[i + 1] == "honeysuckle":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "rhyolitic" and words[i + 1] == "magma":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "monarch" and words[i + 1] == "butterflies":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "fish" and words[i + 1] == "species":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "amur" and words[i + 1] == "honeysuckle":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "recycled" and words[i + 1] == "crust":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "pulse" and words[i + 1] == "warming":
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1

                   
            elif words[i] == "gem" and words[i + 1].startswith("diamond"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "whale" and words[i + 1].startswith("population"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1                 
            elif words[i] == "carbon" and words[i + 1].startswith("sink"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "vertebrate" and words[i + 1].startswith("population"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "ocean" and words[i + 1].startswith("acidification"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "sea" and words[i + 1].startswith("level"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "plastic" and words[i + 1].startswith("contaminant"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "sediment" and words[i + 1].startswith("core"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "magnetic" and words[i + 1].startswith("storm"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "permafrost" and words[i + 1].startswith("thaw"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "sea" and words[i + 1].startswith("lion"):
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
                  
                   
            elif words[i] == "migratory" and words[i + 1].startswith("bird"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "air" and words[i + 1].startswith("temperature"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1   
            elif words[i] == "land" and words[i + 1].startswith("temperature"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1                    
                 
            elif words[i] == "sea" and words[i + 1].startswith("current"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "abyssal" and words[i + 1].startswith("seamount"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "iron" and words[i + 1].startswith("isotope"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1 
            elif words[i] == "surface" and words[i + 1].startswith("water"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "meltwater" and words[i + 1].startswith("peak"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "freshwater" and words[i + 1].startswith("resource"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1    
            elif words[i] == "water" and words[i + 1].startswith("tower"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1           
            elif words[i] == "coral" and words[i + 1].startswith("reef"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict["coral reef"] += 1  
            elif words[i] == "coral" and words[i + 1].startswith("bleach"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1      
            elif words[i] == "cattle" and words[i + 1].startswith("annotation"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1     
            elif words[i] == "hunting" and words[i + 1] == "records": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "fence" and words[i + 1] == "encounters": 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i].startswith("fish") and words[i + 1].startswith("tag"): 
                theme_dict["fish tag"] += 1
            elif words[i].startswith("bird") and words[i + 1].startswith("spec"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i].startswith("detect") and words[i + 1].startswith("animal"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i].startswith("wildlife") and words[i + 1].startswith("monitor"): 
                theme_dict["wildlife monitor"] += 1
            elif words[i].startswith("deepfake") and words[i + 1].startswith("geography"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "krill" and words[i + 1].startswith("fish"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "marine" and words[i + 1].startswith("area"): 
                theme_dict["marine area"] += 1
            elif words[i] == "methane" and words[i + 1].startswith("emission"): 
                theme_dict["methane emission"] += 1
            elif words[i] == "volcanic" and words[i + 1].startswith("eruption"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i] == "envelope-like" and words[i + 1].startswith("region"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
                               

                 
            elif words[i] == "satellite" and words[i + 1].startswith("imagery"): 
                phrase = " ".join(words[i: min(len(words), i + 2)])
                theme_dict[phrase] += 1
            elif words[i].startswith("satellite"): 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i].startswith("biodiversity"): 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "cave-rail": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "cave-rails": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "stopover-to-passage": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "meltwater-rich": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "meltwater": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "water-fetching":
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            
                
            elif words[i] == "cattle": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "mislabeling": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "mantle": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "biosphere": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "microbial": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "precipitation": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "freshwater": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] == "cymothoid": 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i].startswith("shipwreck"): 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
                

            elif words[i] in ["seal", "seals"]: 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] in ["borehole", "boreholes"]: 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] in ["moth", "moths"]: 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
                
            elif words[i] in ["primate", "primates"]: 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] in ["inclusion", "inclusions"]: 
                phrase = " ".join(words[i: min(len(words), i + 1)])
                theme_dict[phrase] += 1
            elif words[i] in ["extinct", "extinctive", "extinction"]: 
                theme_dict["extinct"] += 1

                
            elif words[i] == "water" and words[i + 1] == "storage" and words[i + 2] == "capacity":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "ice" and words[i + 1] == "mass" and words[i + 2] == "loss":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "antarctic" and words[i + 1] == "Antarctic" and words[i + 2] == "sheet":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "marine" and words[i + 1] == "plastic" and words[i + 2] == "litter":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "human" and words[i + 1] == "activity" and words[i + 2] == "records":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict["human activity records"] += 1
            elif words[i] == "oil" and words[i + 1] == "and" and words[i + 2] == "gas":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "shark" and words[i + 1] == "core" and words[i + 2] == "regions":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "oil" and words[i + 1] == "and" and words[i + 2] == "gas":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "oil" and words[i + 1] == "palm" and words[i + 2] == "cultivation":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "arctic" and words[i + 1] == "sea" and words[i + 2] == "ice":
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "glass" and words[i + 1] == "sponge" and words[i + 2].startswith("reef"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "freshwater" and words[i + 1] == "export" and words[i + 2].startswith("increase"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "sea" and words[i + 1] == "ice" and words[i + 2].startswith("decline"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1 
            elif words[i] == "glass" and words[i + 1] == "sponge" and words[i + 2].startswith("reef"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "vertical" and words[i + 1] == "tracking" and words[i + 2].startswith("microscope"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "invasive" and words[i + 1] == "weed" and words[i + 2].startswith("species"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "reef" and words[i + 1] == "fish" and words[i + 2].startswith("evolution"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "antarctic" and words[i + 1] == "ice" and words[i + 2].startswith("sheet"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            elif words[i] == "greenland" and words[i + 1] == "ice" and words[i + 2].startswith("sheet"):
                phrase = " ".join(words[i: min(len(words), i + 3)])
                theme_dict[phrase] += 1
            
              
        # Check if any form of "bird" and any form of "bird" exist in the document
        text_lower = text.lower()
        if bird_pattern.search(text_lower):
            theme_dict["bird"] += 1
        if flood_pattern.search(text_lower):
            theme_dict["flood"] += 1

            
        elif seal_pattern.search(text_lower):
            theme_dict["seal"] += 1
        elif crater_pattern.search(text_lower):
            theme_dict["crater"] += 1
        elif stopover_passage_pattern.search(text_lower):
            theme_dict["stopover-to-passage"] += 1



        # Save the dictionary as JSON
        fout.write(json.dumps(theme_dict) + '\n')

# Print results
print("Extracted theme_dict occurrences:", theme_dict)