import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy
import random
from typing import List, Union
from pathlib import Path
import torch
from transformer_lens import HookedTransformer


### INDIRECT OBJECT IDENTIFICATION TASK ###

# print("Indirect object identification...")

# NAMES = [
#     "Michael",
#     "Christopher",
#     "Jessica",
#     "Matthew",
#     "Ashley",
#     "Jennifer",
#     "Joshua",
#     "Amanda",
#     "Daniel",
#     "David",
#     "James",
#     "Robert",
#     "John",
#     "Joseph",
#     "Andrew",
#     "Ryan",
#     "Brandon",
#     "Jason",
#     "Justin",
#     "Sarah",
#     "William",
#     "Jonathan",
#     "Stephanie",
#     "Brian",
#     "Nicole",
#     "Nicholas",
#     "Anthony",
#     "Heather",
#     "Eric",
#     "Elizabeth",
#     "Adam",
#     "Megan",
#     "Melissa",
#     "Kevin",
#     "Steven",
#     "Thomas",
#     "Timothy",
#     "Christina",
#     "Kyle",
#     "Rachel",
#     "Laura",
#     "Lauren",
#     "Amber",
#     "Brittany",
#     "Danielle",
#     "Richard",
#     "Kimberly",
#     "Jeffrey",
#     "Amy",
#     "Crystal",
#     "Michelle",
#     "Tiffany",
#     "Jeremy",
#     "Benjamin",
#     "Mark",
#     "Emily",
#     "Aaron",
#     "Charles",
#     "Rebecca",
#     "Jacob",
#     "Stephen",
#     "Patrick",
#     "Sean",
#     "Erin",
#     "Jamie",
#     "Kelly",
#     "Samantha",
#     "Nathan",
#     "Sara",
#     "Dustin",
#     "Paul",
#     "Angela",
#     "Tyler",
#     "Scott",
#     "Katherine",
#     "Andrea",
#     "Gregory",
#     "Erica",
#     "Mary",
#     "Travis",
#     "Lisa",
#     "Kenneth",
#     "Bryan",
#     "Lindsey",
#     "Kristen",
#     "Jose",
#     "Alexander",
#     "Jesse",
#     "Katie",
#     "Lindsay",
#     "Shannon",
#     "Vanessa",
#     "Courtney",
#     "Christine",
#     "Alicia",
#     "Cody",
#     "Allison",
#     "Bradley",
#     "Samuel",
# ]

# ABC_TEMPLATES = [
#     "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
#     "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
#     "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
#     "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
# ]

# BAC_TEMPLATES = [
#     template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
#     for template in ABC_TEMPLATES
# ]

# BABA_TEMPLATES = [
#     "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
#     "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
#     "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
#     "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
#     "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
#     "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
#     "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
#     "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
#     "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
#     "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
#     "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
# ]

# BABA_LONG_TEMPLATES = [
#     "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
#     "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
#     "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
#     "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
#     "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
#     "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
#     "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
#     "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
#     "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
#     "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
#     "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
# ]

# BABA_LATE_IOS = [
#     "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
#     "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
#     "Then, [B] and [A] had a long argument and after that [B] said to [A]",
#     "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
#     "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
# ]

# BABA_EARLY_IOS = [
#     "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
#     "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
#     "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
#     "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
#     "Then [B] and [A] had a long argument, and after that [B] said to [A]",
#     "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
#     "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
#     "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
# ]

# TEMPLATES_VARIED_MIDDLE = [
#     "",
# ]

# ABBA_TEMPLATES = BABA_TEMPLATES[:]
# ABBA_LATE_IOS = BABA_LATE_IOS[:]
# ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

# for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS]:
#     for i in range(len(TEMPLATES)):
#         first_clause = True
#         for j in range(1, len(TEMPLATES[i]) - 1):
#             if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
#                 TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
#             elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
#                 first_clause = False
#                 TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

# VERBS = [" tried", " said", " decided", " wanted", " gave"]
# PLACES = [
#     "store",
#     "garden",
#     "restaurant",
#     "school",
#     "hospital",
#     "office",
#     "house",
#     "station",
# ]
# OBJECTS = [
#     "ring",
#     "kiss",
#     "bone",
#     "basketball",
#     "computer",
#     "necklace",
#     "drink",
#     "snack",
# ]

# ANIMALS = [
#     "dog",
#     "cat",
#     "snake",
#     "elephant",
#     "beetle",
#     "hippo",
#     "giraffe",
#     "tiger",
#     "husky",
#     "lion",
#     "panther",
#     "whale",
#     "dolphin",
#     "beaver",
#     "rabbit",
#     "fox",
#     "lamb",
#     "ferret",
# ]

# def multiple_replace(dict, text):
#     # from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
#     # Create a regular expression from the dictionary keys
#     regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

#     # For each match, look-up corresponding value in dictionary
#     return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


# def iter_sample_fast(iterable, samplesize):
#     results = []
#     # Fill in the first samplesize elements:
#     try:
#         for _ in range(samplesize):
#             results.append(next(iterable))
#     except StopIteration:
#         raise ValueError("Sample larger than population.")
#     random.shuffle(results)  # Randomize their positions

#     return results

# NOUNS_DICT = NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}

# def gen_prompt_counterfact(
#     tokenizer,
#     templates, names, nouns_dict, N
# ):
#     nb_gen = 0
#     ioi_prompts = []
#     ioi_prompts_counterfact = []
#     for nb_gen in range(N):
#         temp = rd.choice(templates)
#         temp_id = templates.index(temp)
#         name_1 = ""
#         name_2 = ""
#         name_3 = ""
        
#         # Ensure name_1 and name_2 are different
#         while name_1 == name_2:
#             name_1 = rd.choice(names)
#             name_2 = rd.choice(names)
#             if len(tokenizer(" " + name_1)["input_ids"]) != 1 or len(tokenizer(" " + name_2)["input_ids"]) != 1:
#                 name_1 = ""
#                 name_2 = ""

#         # Ensure name_3 is different from name_1 and name_2 for the counterfactual prompts
#         while name_3 in {name_1, name_2} or name_3 == "":
#             name_3 = rd.choice(names)
#             if len(tokenizer(" " + name_3)["input_ids"]) != 1:
#                 name_3 = ""
                
#         assert all([len(tokenizer(" " + name)["input_ids"]) == 1 for name in [name_1, name_2, name_3]])

#         nouns = {}
#         ioi_prompt = {}
#         ioi_prompt_counterfact = {}
#         for k in nouns_dict:
#             nouns[k] = rd.choice(nouns_dict[k])
#             ioi_prompt[k] = nouns[k]
#             ioi_prompt_counterfact[k] = nouns[k]
#         prompt = temp
#         for k in nouns_dict:
#             prompt = prompt.replace(k, nouns[k])

#         prompt1 = prompt.replace("[A]", name_1)
#         prompt1 = prompt1.replace("[B]", name_2)
#         ioi_prompt["text"] = prompt1
#         ioi_prompt["IO"] = name_1
#         ioi_prompt["S"] = name_2
#         ioi_prompt["TEMPLATE_IDX"] = temp_id
#         ioi_prompts.append(ioi_prompt)

#         prompt2 = prompt.replace("[A]", name_3)
#         prompt2 = prompt2.replace("[B]", name_2)
#         # Replace the second occurrence of name_3 with name_1
#         prompt2 = prompt2.replace(name_2, name_1, 1)
#         ioi_prompt_counterfact["text"] = prompt2
#         ioi_prompt_counterfact["IO"] = name_3
#         ioi_prompt_counterfact["S"] = name_2
#         ioi_prompt_counterfact["TEMPLATE_IDX"] = temp_id
#         ioi_prompts_counterfact.append(ioi_prompt_counterfact)
    
#     return ioi_prompts, ioi_prompts_counterfact

# def gen_ioi_dataset(
#     tokenizer,
#     n_prompts,
# ):
#     assertion = False
#     while not assertion:
#         prompts, prompts_cf = gen_prompt_counterfact(
#             tokenizer,
#             ABBA_TEMPLATES + BABA_TEMPLATES,
#             NAMES,
#             NOUNS_DICT,
#             n_prompts,
#         )

#         prompts = [prompt["text"] for prompt in prompts]
#         prompts_cf = [prompt["text"] for prompt in prompts_cf]

#         # ignore final token (indirect object)
#         prompts = tokenizer(prompts)["input_ids"]
#         prompts_cf = tokenizer(prompts_cf)["input_ids"]

#         assertion = all([len(prompt) == len(prompt_cf) for prompt, prompt_cf in zip(prompts, prompts_cf)])

#     # calc seq lengths & pad
#     seq_lengths = torch.tensor([len(prompt)-1 for prompt in prompts])
#     max_seq_length = torch.max(seq_lengths)

#     prompts = torch.stack(
#         [torch.tensor(prompt[:-1] + [0]*(max_seq_length - len(prompt[:-1]))) for prompt in prompts]
#     )

#     prompts_cf = torch.stack(
#         [torch.tensor(prompt[:-1] + [0]*(max_seq_length - len(prompt[:-1]))) for prompt in prompts_cf]
#     )

#     return prompts, prompts_cf, seq_lengths

# def setup_tokenizer(model_name='gpt2'):
#     """
#     Initialize and return a tokenizer based on the specified model.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return tokenizer

# def main(n_prompts=100):
#     # Set up the tokenizer
#     tokenizer = setup_tokenizer()
    
#     # Generate the dataset
#     prompts, prompts_cf, seq_lengths = gen_ioi_dataset(tokenizer, n_prompts)

#     return prompts, prompts_cf, seq_lengths

# Set up the model
def prompt_to_resid_stream(prompt: str, model: HookedTransformer, resid_type: str = 'accumulated', position: str = 'last') -> torch.Tensor:
    """
    Convert a prompt to a residual stream of size (n_layers, d_model)
    """
    # Run the model over the prompt
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)

        # Get the accumulated residuals
        if resid_type == 'accumulated':
            resid, _ = cache.accumulated_resid(return_labels=True, apply_ln=True)
        elif resid_type == 'decomposed':
            resid, _ = cache.decompose_resid(return_labels=True)
        elif resid_type == 'heads':
            cache.compute_head_results()
            head_resid, head_labels = cache.stack_head_results(return_labels=True)
            #mlp_resid, mlp_labels = cache.decompose_resid(mode='mlp', incl_embeds=False, return_labels=True)
            # Combine
            # resid = torch.cat([head_resid, mlp_resid], dim=0)
            # labels = head_labels + mlp_labels
            resid = head_resid
            labels = head_labels
        else:
            raise ValueError("resid_type must be one of 'accumulated', 'decomposed', 'heads'")

    # POSITION
    if position == 'last':
        last_token_accum = resid[:, 0, -1, :]  # layer, batch, pos, d_model
    elif position == 'mean':
        last_token_accum = resid.mean(dim=2).squeeze()
    else:
        raise ValueError("position must be one of 'last', 'mean'")
    return last_token_accum, labels

# def all_prompts_to_resid_streams(prompts, prompts_cf, model, resid_type='accumulated', position='mean'):
#     """
#     Convert all prompts and counterfactual prompts to residual streams
#     """
#     # Stack prompts and prompts cf
#     resid_streams, final_prompts = [], []
#     all_prompts = torch.cat([prompts, prompts_cf], dim=0)
#     for i in tqdm(range(all_prompts.shape[0])):
#         prompt = tokenizer.decode(all_prompts[i])
#         # Strip the prompt of any exclamation marks
#         prompt = prompt.replace("!", "")
#         resid_stream, labels = prompt_to_resid_stream(prompt, model, resid_type, position)
#         resid_streams.append(resid_stream)
#     # Stack the residual streams into a single tensor
#     return torch.stack(resid_streams), labels
    
# n_prompts = 250
# device = "cpu" #utils.get_device()
# model = HookedTransformer.from_pretrained("gpt2-small", device=device)
# tokenizer = setup_tokenizer()
# prompts, prompts_negative, seq_lengths = main(n_prompts=n_prompts)

# resid_streams, labels = all_prompts_to_resid_streams(prompts, prompts_negative, model, resid_type='heads', position='mean')

# ground_truth = [
#     (2, 2), (4, 11), # previous token heads
#     (0, 1), (3, 0), (0, 10), # duplicate token heads
#     (5, 5), (6, 9), (5, 8), (5, 9), # induction heads
#     (7, 3), (7, 9), (8, 6), (8, 10), # S-inhibition heads
#     (10, 7), (11, 10), # negative name-mover heads
#     (9, 9), (9, 6), (10, 0), # name-mover heads
#     (10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9), (11, 3), (9, 7), # backup name-mover heads
# ]

# # Save
# torch.save(resid_streams, "../data/ioi/resid_heads_mean.pt")
# all_prompts = model.to_string(torch.cat([prompts, prompts_negative], dim=0))
# torch.save(all_prompts, "../data/ioi/prompts_heads_mean.pt")
# torch.save(ground_truth, "../data/ioi/ground_truth.pt")
# torch.save(labels, "../data/ioi/labels_heads_mean.pt")

# print("Indirect object identification task done.\n\n")

### GREATER-THAN TASK ###

print("Greater-than task...")

device = 'cpu'
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

def generate_real_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {year} to the year {century}"
    if eos:
        sentence = " " + sentence
    return sentence

def real_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 YY to the year XX2".split()
    if eos:
        sentence = [""] + sentence
    return sentence

def generate_bad_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {century}01 to the year {century}"
    #sentence = f"The {noun} lasted from the year {year} to the year {century-1}"
    if eos:
        sentence = " " + sentence
    return sentence

def bad_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 01 to the year XX2".split()
    if eos:
        sentence = [""] + sentence
    return sentence

def is_valid_year(year: str, model) -> bool:
    _year = " " + year
    token = model.to_tokens(_year)
    detok = model.to_string(token)
    return len(detok) == 2 and len(detok[1]) == 2

class YearDataset:
    years_to_sample_from: torch.Tensor
    N: int
    ordered: bool
    eos: bool

    nouns: List[str]
    years: torch.Tensor
    years_YY: torch.Tensor
    good_sentences: List[str]
    bad_sentences: List[str]
    good_toks: torch.Tensor
    bad_toks: torch.Tensor
    good_prompt: List[str]
    bad_prompt: List[str]
    good_mask: torch.Tensor
    model: HookedTransformer

    def __init__(
        self,
        years_to_sample_from,
        N: int,
        nouns: Union[str, List[str], Path],
        model: HookedTransformer,
        balanced: bool = True,
        eos: bool = False,
        device: str = "cpu",
    ):
        self.years_to_sample_from = years_to_sample_from
        self.N = N
        self.eos = eos
        self.model = model

        if isinstance(nouns, str):
            noun_list = [nouns]
        elif isinstance(nouns, list):
            noun_list = nouns
        elif isinstance(nouns, Path):
            with open(nouns, "r") as f:
                noun_list = [line.strip() for line in f]
        else:
            raise ValueError(f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}")

        self.nouns = random.choices(noun_list, k=N)

        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(N):
                sample_pool = self.years_to_sample_from[years_to_sample_from_YY == current_year]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            self.years = torch.tensor(years)
        else:
            self.years = torch.tensor(self.years_to_sample_from[torch.randint(0, len(self.years_to_sample_from), (N,))])

        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        self.good_sentences = [
            generate_real_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]
        self.bad_sentences = [
            generate_bad_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]

        self.good_prompt = real_sentence_prompt(eos=eos)
        self.bad_prompt = bad_sentence_prompt(eos=eos)

    def __len__(self):
        return self.N

# Instantiate the model
model = HookedTransformer.from_pretrained("gpt2-small", device='cpu')

# Define your nouns and years
nouns = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
]  # Example nouns list
years_to_sample_from = torch.arange(1200, 2000)  # Example years range

# Instantiate the YearDataset class
dataset = YearDataset(
    years_to_sample_from=years_to_sample_from,
    N=250,  # Number of samples you want
    nouns=nouns,
    model=model,
    balanced=True,  # Whether to balance the years in the dataset
    eos=False,  # Whether to add an end-of-sentence token
    device="cpu"  # Device to use ('cpu' or 'cuda')
)


# def all_prompts_to_resid_streams_gt(prompts, prompts_cf, model, resid_type='accumulated', position='mean'):
#     """
#     Convert all prompts and counterfactual prompts to residual streams
#     """
#     # Stack prompts and prompts cf
#     resid_streams = []
#     # Combine the lists of strs
#     all_prompts = prompts + prompts_cf
#     for i in tqdm(range(len(all_prompts))):
#         prompt = all_prompts[i]
#         resid_stream, labels = prompt_to_resid_stream(prompt, model, resid_type, position)
#         resid_streams.append(resid_stream)
#     # Stack the residual streams into a single tensor
#     return torch.stack(resid_streams), labels

# prompts = dataset.good_sentences
# prompts_cf = dataset.bad_sentences

# resid_streams, labels = all_prompts_to_resid_streams_gt(prompts, prompts_cf, model, resid_type='heads', position='mean')

# ground_truth = [
#     (0, 3), (0, 5), (0, 1), (5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1), # attention heads
# ]

# # Save to ..data/gt folder
# torch.save(resid_streams, "../data/gt/resid_heads_mean.pt")
# all_prompts = prompts + prompts_cf
# torch.save(all_prompts, "../data/gt/prompts_heads_mean.pt")
# torch.save(labels, "../data/gt/labels_heads_mean.pt")
# torch.save(ground_truth, "../data/gt/ground_truth.pt")


# print("Greater-than task done.\n\n")


