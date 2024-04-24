from typing import Union, List
import random
import random
from typing import List, Union
from pathlib import Path
import torch
from transformer_lens import HookedTransformer

from data_utils import all_prompts_to_resid_streams_gt


### GREATER-THAN TASK ###

print("Greater-than task...")

device = "cpu"
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
    # sentence = f"The {noun} lasted from the year {year} to the year {century-1}"
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
            raise ValueError(
                f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}"
            )

        self.nouns = random.choices(noun_list, k=N)

        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(N):
                sample_pool = self.years_to_sample_from[
                    years_to_sample_from_YY == current_year
                ]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            self.years = torch.tensor(years)
        else:
            self.years = torch.tensor(
                self.years_to_sample_from[
                    torch.randint(0, len(self.years_to_sample_from), (N,))
                ]
            )

        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        self.good_sentences = [
            generate_real_sentence(noun, int(year.item()), eos=eos)
            for noun, year in zip(self.nouns, self.years)
        ]
        self.bad_sentences = [
            generate_bad_sentence(noun, int(year.item()), eos=eos)
            for noun, year in zip(self.nouns, self.years)
        ]

        self.good_prompt = real_sentence_prompt(eos=eos)
        self.bad_prompt = bad_sentence_prompt(eos=eos)

    def __len__(self):
        return self.N


# Instantiate the model
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

# Define your nouns and years
nouns = [
    "abduction",
    "accord",
    "affair",
    "agreement",
    "appraisal",
    "assaults",
    "assessment",
    "attack",
    "attempts",
    "campaign",
    "captivity",
    "case",
    "challenge",
    "chaos",
    "clash",
    "collaboration",
    "coma",
    "competition",
    "confrontation",
    "consequence",
    "conspiracy",
    "construction",
    "consultation",
    "contact",
    "contract",
    "convention",
    "cooperation",
    "custody",
    "deal",
    "decline",
    "decrease",
    "demonstrations",
    "development",
    "disagreement",
    "disorder",
    "dispute",
    "domination",
    "dynasty",
    "effect",
    "effort",
    "employment",
    "endeavor",
    "engagement",
    "epidemic",
    "evaluation",
    "exchange",
    "existence",
    "expansion",
    "expedition",
    "experiments",
    "fall",
    "fame",
    "flights",
    "friendship",
    "growth",
    "hardship",
    "hostility",
    "illness",
    "impact",
    "imprisonment",
    "improvement",
    "incarceration",
    "increase",
    "insurgency",
    "invasion",
    "investigation",
    "journey",
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
    device="cpu",  # Device to use ('cpu' or 'cuda')
)

prompts = dataset.good_sentences
prompts_cf = dataset.bad_sentences

resid_streams, labels = all_prompts_to_resid_streams_gt(
    prompts, prompts_cf, model, resid_type="heads", position="mean"
)

ground_truth = [
    (0, 3),
    (0, 5),
    (0, 1),
    (5, 5),
    (6, 1),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 1),  # attention heads
]

# Save to ..data/gt folder
torch.save(resid_streams, "../data/gt/resid_heads_mean.pt")
all_prompts = prompts + prompts_cf
torch.save(all_prompts, "../data/gt/prompts_heads_mean.pt")
torch.save(labels, "../data/gt/labels_heads_mean.pt")
torch.save(ground_truth, "../data/gt/ground_truth.pt")


print("Greater-than task done.\n\n")
