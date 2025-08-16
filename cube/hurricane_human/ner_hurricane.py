import spacy
from collections import defaultdict
from tqdm import tqdm
import json
from collections import Counter
from IPython import embed
import re

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_trf")  


# read date_source
date_list = []
with open('corpus/hurricane/SciDCC-Hurricane.txt') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        date_list.append(line.strip())
print(f'Length of date_list: {len(date_list)}')


# ================ Process whole documents and extract for one ENTITY ================
with open('hypercube/hurricane/date.txt','w') as fout:
    for t in tqdm(date_list):
        text = (t)
        doc = nlp(text)
        entity_dict = defaultdict(int)
        print("entity_dict:", entity_dict)

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            if entity.label_ in ['DATE']:  # ['PERSON', "DATE", "EVENT"]
                entity_dict[entity.text] += 1
        fout.write(json.dumps(entity_dict) + '\n')




# # ================ Process whole documents and extract for MERGED ENTITY ================
# ENTITY_MAPPING = {
#     "GPE": "LOCATION",  # Merge GPE into LOCATION
#     "LOC": "LOCATION",  # Merge LOC into LOCATION
# }

# # Process documents and extract entities
# with open('hypercube/hurricane/location.txt', 'w') as fout:
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



# # ================ theme ================
# hurr_pattern = re.compile(r'\bhurricane(s)?\b', re.IGNORECASE)  # Matches "hurricane" or "hurricanes"
# landfall_pattern = re.compile(r'\blandfall(s|d|ing)?\b', re.IGNORECASE)  # Matches "landfall", "landfalls", "landfalling", "landfalled"
# indian_pattern = re.compile(r'\bindian?\b', re.IGNORECASE)  
# monsoon_pattern = re.compile(r'\bmonsoon(s|d|ing)?\b', re.IGNORECASE)
# path_pattern = re.compile(r'\bpath(s)?\b', re.IGNORECASE)
# storm_pattern = re.compile(r'\bstorm(s)?\b', re.IGNORECASE)
# micro_pattern = re.compile(r'\bmicrophone(s)?\b', re.IGNORECASE)
# disaster_pattern = re.compile(r'\bdisaster(s)?\b', re.IGNORECASE)
# flood_pattern = re.compile(r'\bflood(s|ing)?\b', re.IGNORECASE)

# # Process the text and count occurrences
# with open('hypercube/hurricane/theme.txt', 'w') as fout:
#     for text in tqdm(date_list):
#         theme_dict = Counter()

#         doc = nlp(text)
#         # Find named entities, phrases and concepts
#         for entity in doc.ents:
#             text_data = entity.text
#             if text_data.startswith('El') and entity.label_ in ['EVENT']: 
#                 theme_dict["El Niño"] += 1
#             if text_data.startswith('La') and entity.label_ in ['EVENT']: 
#                 theme_dict["La Niña"] += 1
        
#         # Capture continuous two words, e.g., "climate change", "indian monsoons", 
#         words = text.lower().split()
#         for i, word in enumerate(words):
#             if words[i] == "climate" and words[i + 1] == "change":
#                 # phrase = " ".join(words[max(0, i - 2): min(len(words), i + 3)])  # Capture 2 words before and after
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "climate" and words[i + 1] == "variability":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "indian" and words[i + 1] == "monsoons":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "greenhouse" and words[i + 1] == "gas":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "hurricane" and words[i + 1] == "season":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "water" and words[i + 1] == "quality":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "hurricane" and words[i + 1] == "intensity":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "storm" and words[i + 1] == "intensity":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "sooty" and words[i + 1] == "terns":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "carbon" and words[i + 1] == "uptake":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "kinetic" and words[i + 1] == "energy":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "global" and words[i + 1] == "warming":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "hurricane" and words[i + 1] == "hitting":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "hurricane" and words[i + 1] == "monitoring":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "african" and words[i + 1] == "dust":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             # elif words[i] == "energy" and words[i + 1] == "source":
#             #     phrase = " ".join(words[i: min(len(words), i + 2)]) 
#             #     theme_dict[phrase] += 1
#             elif words[i] == "sea" and words[i + 1] == "surface" and words[i + 2] == "temperature":
#                 phrase = " ".join(words[i: min(len(words), i + 3)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "intense" and words[i + 1] == "hurricanes":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "major" and words[i + 1] == "hurricanes":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "El" and words[i + 1] in ["Niño", "Nino"]:
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "La" and words[i + 1] in ["Niña", "Nina"]:
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "human" and words[i + 1] == "mobility":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "public" and words[i + 1] == "health":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1                 
#             elif words[i] == "climate" and words[i + 1] == "models":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "hurricane" and words[i + 1] == "prediction":
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "household" and words[i + 1] == "respondents":
#                 phrase = " ".join(words[i: min(len(words), i + 2)])
#                 theme_dict[phrase] += 1
#             elif words[i] == "climate" and words[i + 1] == "simulations":
#                 phrase = " ".join(words[i: min(len(words), i + 2)])
#                 theme_dict[phrase] += 1
#             elif words[i] == "anxiety" and words[i + 1] == "levels":
#                 phrase = " ".join(words[i: min(len(words), i + 2)])
#                 theme_dict[phrase] += 1       
#             elif words[i] == "storm-force" and words[i + 1] == "wind":
#                 phrase = " ".join(words[i: min(len(words), i + 2)])
#                 theme_dict[phrase] += 1
#             elif words[i] == "wind" and words[i + 1] == "speed":
#                 phrase = " ".join(words[i: min(len(words), i + 2)])
#                 theme_dict[phrase] += 1
#             elif words[i] == "global" and words[i + 1].startswith("hawk"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "economic" and words[i + 1].startswith("loss"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "central" and words[i + 1].startswith("pressure"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "sustained" and words[i + 1].startswith("wind"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "tropical-storm-force" and words[i + 1].startswith("wind"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
                   
#             elif words[i] == "tropical" and words[i + 1].startswith("cyclone"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "named" and words[i + 1].startswith("storm"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "major" and words[i + 1].startswith("hurricane"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "MM5" and words[i + 1].startswith("experiment"): 
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "MM5" and words[i + 1].startswith("simulation"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "storm" and words[i + 1].startswith("simulation"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "storm" and words[i + 1].startswith("surge"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "climate" and words[i + 1].startswith("condition"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "rapid" and words[i + 1].startswith("cooling"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "upper-level" and words[i + 1].startswith("trough"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "kidney" and words[i + 1].startswith("disease"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "coral" and words[i + 1].startswith("reef"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "coral" and words[i + 1].startswith("bleach"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "ocean" and words[i + 1].startswith("temperature"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "wind" and words[i + 1].startswith("tower"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "damaging" and words[i + 1].startswith("wind"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "climate" and words[i + 1].startswith("extreme"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "human" and words[i + 1].startswith("health"):
#                 phrase = " ".join(words[i: min(len(words), i + 2)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "human" and words[i + 1] == "and" and words[i + 2] == "economic" and words[i + 3].startswith("cost"):
#                 phrase = " ".join(words[i: min(len(words), i + 4)]) 
#                 theme_dict[phrase] += 1
#             elif words[i] == "GDP": 
#                 theme_dict["GDP"] += 1
#             elif words[i] == "paleohurricane":
#                 theme_dict["paleohurricane"] += 1
#             elif words[i] in ["storm", "storms"]: 
#                 theme_dict["storm"] += 1
#             elif words[i] in ["cyclone", "cyclones", "tropical cyclone", "tropical cyclones"]:
#                 theme_dict["cyclone"] += 1
#             elif words[i] in ["urbanization", "urbanizations"]:
#                 theme_dict["urbanization"] += 1
#             elif words[i] in ["rainfall", "rainfalls"]:
#                 theme_dict["rainfall"] += 1


#         # Check if any form of "hurricane" and any form of "landfall" exist in the document
#         text_lower = text.lower()
#         if hurr_pattern.search(text_lower) and landfall_pattern.search(text_lower):
#             theme_dict["hurricane landfall"] += 1
#         elif hurr_pattern.search(text_lower) and path_pattern.search(text_lower) and indian_pattern.search(text_lower) and monsoon_pattern.search(text_lower):
#             theme_dict["hurricane paths"] += 1
#         elif micro_pattern.search(text_lower):
#             theme_dict["microphone"] += 1
#         elif disaster_pattern.search(text_lower):
#             theme_dict["disaster"] += 1
#         elif flood_pattern.search(text_lower):
#             theme_dict["flood"] += 1
#         elif "antenna" in text_lower or "antennas" in text_lower:
#             theme_dict["antenna"] += 1
#         elif "hot tower" in text_lower or "hot towers" in text_lower:
#             theme_dict["hot tower"] += 1
#         elif "oil and gas" in text_lower:
#             theme_dict["oil and gas"] += 1
#         elif "heavy rain" in text_lower:
#             theme_dict["heavy rain"] += 1
#         elif "ocean-observing system" in text_lower:
#             theme_dict["ocean-observing system"] += 1
#         elif "human and economic costs" in text_lower:
#             theme_dict["human and economic cost"] += 1
#         elif "mental health" in text_lower:
#             theme_dict["mental health"] += 1
#         elif "fisheries" in text_lower:
#             theme_dict["fisheries"] += 1

#         # Save the dictionary as JSON
#         fout.write(json.dumps(theme_dict) + '\n')

# # Print results
# print("Extracted theme_dict occurrences:", theme_dict)