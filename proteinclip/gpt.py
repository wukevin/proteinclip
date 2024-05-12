"""
Code for interfacing with ChatGPT API
"""

import os
import re
import logging
import functools
from typing import *

import numpy as np
import openai
from diskcache import FanoutCache

GPT_EMBED_MODELS = Literal[
    "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
]

embedding_cache = FanoutCache(
    directory=os.path.expanduser("~/.cache/openai"),
    timeout=0.1,
    size_limit=int(8e9),  # Size limit of cache in bytes; 8GB
    eviction_policy="least-recently-used",
)
keyword_cache = FanoutCache(
    directory=os.path.expanduser("~/.cache/openai_keywords"),
    timeout=0.1,
    size_limit=int(5e9),  # Size limit of cache in bytes; 5GB
    eviction_policy="least-recently-used",
)


@functools.lru_cache
def get_openai_api_key(
    fname: str = os.path.join(os.path.dirname(__file__), "gpt_key")
) -> str:
    """Return the API key to query OpenAI with."""
    if not os.path.isfile(fname):
        raise ValueError(f"Key file not found: {fname}")
    with open(fname, "r") as source:
        key = source.readlines()
    assert len(key) == 1
    return key.pop().strip()


CLIENT = openai.OpenAI(api_key=get_openai_api_key())


def sanitize_text(ft: str, max_len: int = 5192) -> str:
    """Sanitize a clinical feature string for querying GPT."""
    # Replace the {12: Foo et al. (2001)} curly brackets indicating references
    ft = re.sub(r"\{[0-9]+:", "", ft).replace(
        "}", ""
    )  # Avoids removing actual reference
    # Whitespace fixes
    retval = ft.replace("\n", " ").replace("\t", " ").replace("  ", " ").strip()

    # If too long, truncate it; max length is 8192 tokens.
    words = retval.split(" ")
    if len(words) > max_len:  # A bit of a buffer.
        retval = " ".join(words[:max_len])
    return retval


@embedding_cache.memoize(typed=True, name="gptprotein-embed")
def get_openai_embedding(
    s: str, model: GPT_EMBED_MODELS = "text-embedding-ada-002"
) -> np.ndarray:
    """Get the embeddings for the given string s."""
    s = s.strip()  # Remove leading and trailing whitespace
    embed = CLIENT.embeddings.create(input=[s], model=model).data[0].embedding
    return np.array(embed)


@keyword_cache.memoize(typed=True, name="gptprotein-keywords")
def get_openai_keywords(text: str, addtl_sys_prompt: str = "") -> List[str]:
    """Get the keywords from OpenAI."""

    keywords = CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You will be provided with a block of text, and your task is to extract a list of keywords from it, formatted as a comma-separated list. "
                    + addtl_sys_prompt
                ).strip(),
            },
            {
                "role": "user",
                "content": sanitize_text(text, max_len=2000),
            },
        ],
        temperature=0.5,
        max_tokens=64,
        top_p=1,
    )
    content = keywords.choices[0].message.content
    return [s.strip() for s in content.split(",") if s.strip()]


# https://www.omim.org/entry/100070
SAMPLE_TEXT_1 = """
Loosemore et al. (1988) described 2 brothers with abdominal aortic aneurysm at ages 58 and 62 years, whose father died of ruptured abdominal aortic aneurysm at the age of 72 years. Four other sibs died of myocardial infarction at ages 47 to 61 years. Loosemore et al. (1988) suggested that a deficiency of type III collagen (see 120180) might be the basis for the aneurysm formation. The proportion of type III collagen in forearm skin biopsies was cited as accurately reflective of the proportion in the aorta and was said to have been low in the brothers. 

Ward (1992) looked for association of dilated peripheral arteries with aortic aneurysmal disease by measuring the diameters of the common femoral, popliteal, brachial, common carotid, internal carotid, and external carotid arteries by color-flow duplex scan in 30 control subjects and 36 patients with aortic aneurysm matched for age, sex, smoking habits, and hypertension. Mean peripheral artery diameter was significantly greater in patients with aortic aneurysms than in controls at all measurement sites. Peripheral artery dilatation was identified at sites that are seldom, if ever, involved in atherosclerosis. Ward (1992) concluded that there is a generalized dilating diathesis in aortic aneurysmal disease that may be unrelated to atherosclerosis. 

In the study of Verloes et al. (1995), familial male cases showed a significantly earlier age at rupture and a greater rupture rate as compared with sporadic male cases, as well as a tendency (p less than 0.05) towards earlier age of diagnosis. 

AAA occurs among approximately 1.5% of the male population older than 50 years of age. Several studies have indicated an increased frequency among first-degree relatives of patients with AAA. Aneurysms of the peripheral arteries (femoral, popliteal, and isolated iliac) are less common than aortic aneurysms (Lawrence et al., 1995), and arteriomegaly (diffuse aneurysmal disease) is even less common (Hollier et al., 1983). Peripheral aneurysms and arteriomegaly carry a high risk for complications such as rupture, embolism, or thrombosis.
"""

# https://www.omim.org/entry/143100
SAMPLE_TEXT_2 = """
The classic signs of Huntington disease are progressive chorea, rigidity, and dementia. A characteristic atrophy of the caudate nucleus is seen radiographically. Typically, there is a prodromal phase of mild psychotic and behavioral symptoms which precedes frank chorea by up to 10 years. Chandler et al. (1960) observed that the age of onset was between 30 and 40 years. In a study of 196 kindreds, Reed and Neel (1959) found only 8 in which both parents of a single patient with Huntington chorea were 60 years of age or older and normal. The clinical features developed progressively with severe increase in choreic movements and dementia. The disease terminated in death on average 17 years after manifestation of the first symptoms. 

Folstein et al. (1984, 1985) contrasted HD in 2 very large Maryland pedigrees: an African American family residing in a bayshore tobacco farming community and a white Lutheran family living in a farming community in the western Maryland foothills and descended from an immigrant from Germany. They differed, respectively, in age at onset (33 years vs 50 years), presence of manic-depressive symptoms (2 vs 75), number of cases of juvenile onset (6 vs 0), mode of onset (abnormal gait vs psychiatric symptoms), and frequency of rigidity or akinesia (5/21 vs 1/15). In the African American family, the mean age at onset was 25 years when the father was affected and 41 years when the mother was affected; the corresponding figures in the white family were 49 and 52 years. Allelic mutations were postulated. In another survey in Maryland, Folstein et al. (1987) found that the prevalence of HD among African Americans was equal to that in whites. 

Adams et al. (1988) found that life-table estimates of age of onset of motor symptoms have produced a median age 5 years older than the observed mean when correction for truncated intervals of observation (censoring) was made. The bias of censoring refers to the variable intervals of observation and loss to observation at different ages. For example, gene carriers lost to follow-up, those deceased before onset of disease, and those who had not yet manifested the disease at the time of data collection were excluded from the observed distribution of age at onset. 

Kerbeshian et al. (1991) described a patient with childhood-onset Tourette syndrome (137580) who later developed Huntington disease. 

Shiwach (1994) performed a retrospective study of 110 patients with Huntington disease in 30 families. He found the minimal lifetime prevalence of depression to be 39%. The frequency of symptomatic schizophrenia was 9%, and significant personality change was found in 72% of the sample. The age at onset was highly variable: some showed signs in the first decade and some not until over 60 years of age. 

The results of a study by Shiwach and Norbury (1994) clashed with the conventional wisdom that psychiatric symptoms are a frequent presentation of Huntington disease before the development of neurologic symptoms. They performed a control study of 93 neurologically healthy individuals at risk for Huntington disease. The 20 asymptomatic heterozygotes showed no increased incidence of psychiatric disease of any sort when compared to the 33 normal homozygotes in the same group. However, the whole group of heterozygous and homozygous normal at-risk individuals showed a significantly greater number of psychiatric episodes than did their 43 spouses, suggesting stress from the uncertainty associated with belonging to a family segregating this disorder. Shiwach and Norbury (1994) concluded that neither depression nor psychiatric disorders are likely to be significant preneurologic indicators of heterozygous expression of the disease gene. 
"""

if __name__ == "__main__":
    # https://training.seer.cancer.gov/anatomy/body/review.html
    # print(
    #     get_openai_keywords(
    #         SAMPLE_TEXT_1,
    #         addtl_sys_prompt="Find keywords that specify the major organ systems impacted by this clinical description. Choose as many of the following options as are applicable: musculoskeletal, nervous, endocrine, cardiovascular, lymphatic, respiratory, digestive, urinary, reproductive. Do not use other terms in your response.",
    #     ),
    # )
    e = get_openai_embedding(SAMPLE_TEXT_1)
