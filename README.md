# [Koishi's Day 2024](https://danbooru.donmai.us/wiki_pages/koishi_day): Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing

[[English](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/README.md) | [中文](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/README_ZH.md) | [日本語](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/README_JA.md)]

- Let there be fantasy

**[Update]** APC is accepted to **NeurIPS2024**! 💚

## Introduction [\[Paper\]](https://arxiv.org/abs/2405.07726) [\[NeurIPS Version\]](https://neurips.cc/virtual/2024/poster/94451)

Persona-driven Role-playing (PRP) is so cool that it allows you to build AI characters with several short paragraphs to describe the persona (人设/設定)! However, how to keep the AI character faithfully to **ALL** persona statements is a hard problem. PRP agents always make bunches of mistakes or are always vague about the knowledge they should know. 

![Case](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/case_koishi.png)

The main reason behind this limitation is the lack of a metric that can quantify the global PRP faithfulness. So I decide to do so following this intuition:

![APC](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_koishi.png)

Briefly speaking, whenever a query comes from a user, each persona statement will become an active (relevant to the query) or passive (irrelevant to the query) constraint. To satisfy the active constraint, the response needs to be entailed by the statement **(containing the information in the statement)**. Otherwise, for passive constraints, the response only needs to be not contradicted by them **(not containing information that is incorrect according to the persona statement)**. 

![DPO](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/dpo_koishi.png)

We traverse through all persona statements and see whether their constraints are satisfied or not. We count the number of satisfied constraints, which is used as the metric to evaluate the global PRP faithfulness. This metric is named as Active-Passive-Constraint (APC) score. [Direct preference optimization (DPO)](https://arxiv.org/abs/2305.18290) is a method that can encourage models to perform more like responses preferred by humans or criteria. Thus, we can sample two responses to the same query and then apply DPO based on their APC scores to encourage the PRP agent to be globally more faithful to persona statements. 

In practice, the APC scores are assigned by probabilistic models towards a more accurate estimation, the statement-query relevance probability and the statement-to-response natural language inference probability, formalized as follows,

![Formula](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_formula.png)

If you hate formulas, the only thing you need to know is that we need two probabilistic estimators for **relevance** and **NLI**.

![Distillation](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/distillation_koishi.png)

Thus, we use the pipeline above to build such estimators by distilling from GPT-4 with synthesized datasets. So far, the puzzle for global PRP faithfulness quantification and optimization is completed, let's begin our journey to build faithful PRP agents, and be of good cheer!

## Preparation

Before your journey, you need to prepare the following stuff:

1. Download MiniConda3 following the instructions on this [page](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Create an OpenAI account and get an OpenAI API following the instructions on this [page](https://openai.com/index/openai-api/).

3. Create a Huggingface account and create a Huggingface Token for reading following the instructions on this [page](https://huggingface.co/settings/tokens).

4. My implementation is based on Gemma-1.1-7b-it, so you have to gain access to Gemma models following the instructions on this [page](https://huggingface.co/google/gemma-1.1-7b-it).

Then you can create an environment and install the required Python packages:

```bash
conda create -n apc python=3.8
conda activate apc
python -m pip install -r requirements.txt
```

## Quick Start

### Learning RAG models with APC-based DPO

I have formalized the learning scenario for the most faithful persona-driven role-playing agent as a simple bash command. You only have to replace the ```openai_key``` and ```hf_token``` in ```bash_is_all_you_need.sh``` with your own, and then run
```bash
bash bash_is_all_you_need.sh
```

This script builds an APC-based DPO PRP system with RAG for Alice (detailed in ```wiki```) by default. You can find the LoRA weights of the PRP agent in ```prp_models```, the intermediate datasets in ```statement```, and intermediate discriminators in ```discriminators``` (if you set ```use_pretrained_discriminator``` to ```False```).

You can build this advanced PRP system for any character you like by simply putting a wiki text (paragraphs separated by "\n\n") in ```wiki``` with name ```{character_name}_wiki.txt```. Then replace the ```character``` in the ```bash_is_all_you_need.sh``` and run it. You will find everything you need in the corresponding directories.

For Chinese characters, please use 

```bash
bash bash_is_all_you_need_for_chinese.sh
```

We have optimized the GPU usage for implementation. However, you still need a >40G GPU to run the bash command.

- Hyperparameter Suggestions

```model_engine```: "gpt-4", the prompts are written specifically for GPT-4, using other LLMs might cause bugs. 

```use_pretrained_discriminator```: True, generally enabled to reduce the cost of generating the relevance and NLI dataset. (You still have to generate persona statements and user queries!)

```prp_scale```: "7b", "2b" Gemma model always refuses to do role-playing.

```max_dpo_data```: 100, which builds the DPO dataset generally in one hour for characters with persona statement numbers around 100.

```lora_rank```: >= 32, lower LoRA rank will hurt the role-playing performance.

```rag_top_k```: 4-6, which is shown to perform the best by the analysis.

### Evaluating Responses with APC score

We implement the APC scoring function in ```score.py``` based on the discriminators defined in ```classifier.py```. Using the function ```score_APC```, you can score the expected constraint satisfaction numbers of different responses based on all persona statements, we provide a use case in ```evaluation_example.py```, as shown as follows.

```python
from classifier import Classifier, get_relevance_discriminator, get_nli_discriminator
from score import score_apc, score_APC

relevance_discriminator = get_relevance_discriminator(character=None, statement_query_relevance_dataset=None, relevance_finetune_epoch=None, use_pretrained_discriminator=True)
nli_discriminator = get_nli_discriminator(character=None, statement_to_response_nli_v2_dataset=None, nli_finetune_epoch=None, use_pretrained_discriminator=True)

character = "Komeiji Koishi"
statements = ["Komeiji Koishi lives with her sister, Komeiji Satori.", "Komeiji Koishi lives in Chireiden."]
query = "Where do you live, Koishi?"
responses = ["I live in Chireiden with my sister, Satori!", "I live in Chireiden!", "I live in Hakurei Shrine!"]
print([score_APC(character, statements, query, response, relevance_discriminator, nli_discriminator).item() for response in responses])

# [1.6079180240631104, 0.9955980777740479, 0.03315635025501251]
```

Based on the output scores, you can have a rough understanding of how the APC score views PRP faithfulness.

### Chat with Learned AI Characters!

After running the APC-based DPO, you will get a LoRA weight for your character at ```prp_models/gemma-1.1-7b-it-lora-{character}-rag-dpo```, which can be used for chatting with your AI character. We provide an example in ```chat_example.py```, also shown as follows.

```python
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prp_model import retrieval_augmented_generate
from classifier import get_relevance_discriminator

character = "Your Character"

statements = [data["statement"] for data in json.load(open(f"statement/{character}.json"))]

model_id = f"prp_models/gemma-1.1-7b-it-lora-{character}-rag-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

prp_tokenizer = AutoTokenizer.from_pretrained(model_id)
prp_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

relevance_discriminator = get_relevance_discriminator(character=None, statement_query_relevance_dataset=None, relevance_finetune_epoch=None, use_pretrained_discriminator=True)

print(f"You are chatting with {character}!")

with torch.no_grad():
    
    while True:
    
        _, response = retrieval_augmented_generate(character, statements, input('User: '), prp_model, prp_tokenizer, relevance_discriminator, rag_top_k=5)
        response = character+": "+response.replace("<eos>", "")
        print(response)
```

The following is an example conversation with Komeiji Koishi:

```
User: Hi, Koishi! What is your ability?
Komeiji Koishi: I call it the "Silent Whisperer." It allows me to manipulate the unconsciousness of others, making me invisible and granting me control over their actions.
User: Where do you live?
Komeiji Koishi: The Palace of the Earth Spirits serves as my humble abode.
User: Who is your sister?
Komeiji Koishi: Satori Komeiji. The one with all the serious face. 😜
```

Currently, the system only supports single-turn conversations due to the topic scope discussed in our paper. We will put more engineering effort into supporting multi-turn conversations soon!

## Datasets and Checkpoints

The synthesized dataset for statement-query relevance: [KomeijiForce/role-playing-apc-relevance](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-relevance) (English), [KomeijiForce/role-playing-apc-multilingual-relevance](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-multilingual-relevance) (Multilingual)

The synthesized dataset for statement-to-response NLI: [KomeijiForce/role-playing-apc-nli](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-nli) (English), [KomeijiForce/role-playing-apc-multilingual-nli](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-multilingual-nli) (Multilingual)

The fine-tuned DeBERTa-V3-Large discriminator for statement-query relevance: [KomeijiForce/deberta-v3-large-relevance-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-relevance-12character) (English), [KomeijiForce/xlm-roberta-large-relevance-multilingual-12character](https://huggingface.co/datasets/KomeijiForce/xlm-roberta-large-relevance-multilingual-12character) (Multilingual)

The fine-tuned DeBERTa-V3-Large discriminator for statement-to-response NLI: [KomeijiForce/deberta-v3-large-relevance-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-nli-12character) (English), [KomeijiForce/xlm-roberta-large-nli-multilingual-12character](https://huggingface.co/KomeijiForce/xlm-roberta-large-nli-multilingual-12character) (Multilingual)

### Statistics of the PRP datasets

| Character | Persona Statements | Questions | Relevance Data | NLI Data |
|-----------|--------------------|-----------|----------------|----------|
| Alice     | 8                  | 10        | 64             | 144      |
| Bob       | 19                 | 10        | 152            | 459      |
| Eve       | 30                 | 10        | 240            | 545      |
| Beethoven | 383                | 77        | 3061           | 6774     |
| Newton    | 354                | 90        | 2832           | 6331     |
| Socrates  | 324                | 89        | 2591           | 5760     |
| Spartacus | 77                 | 89        | 616            | 1368     |
| Hermione  | 146                | 118       | 1167           | 2586     |
| Voldemort | 201                | 77        | 1608           | 3546     |
| Cleopatra | 374                | 93        | 2991           | 6660     |
| Caesar    | 498                | 87        | 3981           | 8856     |
| Martin Luther King       | 599                | 92        | 4789           | 10644    |

### Performance

<table>
  <thead>
    <tr>
      <th rowspan="2">Character</th>
      <th rowspan="2">Metric </th>
      <th colspan="3">Alice</th>
      <th colspan="3">Bob</th>
      <th colspan="3">Eve</th>
    </tr>
    <tr>
      <th>ΔAPC (DeB)</th>
      <th>ΔAPC (GPT-4)</th>
      <th>Human</th>
      <th>ΔAPC (DeB)</th>
      <th>ΔAPC (GPT-4)</th>
      <th>Human</th>
      <th>ΔAPC (DeB)</th>
      <th>ΔAPC (GPT-4)</th>
      <th>Human</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" style="writing-mode: vertical-lr; text-align: center;">w/o APC-based DPO</td>
      <td>Gemma-7B</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>1.8</td>
      <td>1.1</td>
      <td>0.4</td>
      <td>1.8</td>
      <td>0.7</td>
      <td>-0.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>EU</td>
      <td>2.6</td>
      <td>1.1</td>
      <td>6.4</td>
      <td>3.4</td>
      <td>1.1</td>
      <td>6.2</td>
      <td>3.6</td>
      <td>0.7</td>
      <td>4.6</td>
    </tr>
    <tr>
      <td>LCM</td>
      <td>2.6</td>
      <td>1.4</td>
      <td>6.8</td>
      <td>4.5</td>
      <td>2.2</td>
      <td>7.2</td>
      <td>3.9</td>
      <td>0.7</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>RAG</td>
      <td>2.8</td>
      <td>1.8</td>
      <td>6.8</td>
      <td>4.0</td>
      <td>1.7</td>
      <td>6.8</td>
      <td>4.8</td>
      <td>2.4</td>
      <td>5.8</td>
    </tr>
    <tr>
      <td rowspan="6" style="writing-mode: vertical-lr; text-align: center;">w/ APC-based DPO</td>
      <td>EU</td>
      <td>2.7 (+0.1)</td>
      <td>1.4 (+0.3)</td>
      <td>6.8 (+0.4)</td>
      <td>3.8 (+0.4)</td>
      <td>1.8 (+0.7)</td>
      <td>6.8 (+0.6)</td>
      <td>3.9 (+0.3)</td>
      <td>0.9 (+0.2)</td>
      <td>5.2 (+0.6)</td>
    </tr>
    <tr>
      <td>LCM</td>
      <td>2.8 (+0.2)</td>
      <td><b>2.2 (+0.8)</b></td>
      <td><b>7.6 (+0.8)</b></td>
      <td><b>5.3 (+0.8)</b></td>
      <td>2.5 (+0.3)</td>
      <td>7.8 (+0.6)</td>
      <td>5.1 (+1.2)</td>
      <td>3.3 (+2.6)</td>
      <td>6.6 (+1.6)</td>
    </tr>
    <tr>
      <td>RAG</td>
      <td><b>2.9 (+0.1)</b></td>
      <td><b>2.2 (+0.4)</b></td>
      <td><b>7.6 (+0.8)</b></td>
      <td>5.2 (+1.2)</td>
      <td><b>3.8 (+2.1)</b></td>
      <td><b>8.2 (+1.2)</b></td>
      <td><b>5.8 (+1.0)</b></td>
      <td><b>4.2 (+1.8)</b></td>
      <td><b>7.0 (+1.2)</b></td>
    </tr>
  </tbody>
</table>




## Todo List

- Support More Languages 5/15: Chinese characters are supported.
- Support Multi-turn Conversations
- Allow more Customized Training Setups

## Citation

```bibtex
@inproceedings{APC,
  author       = {Letian Peng and
                  Jingbo Shang},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/309cadc33589efca4018a490c07db263-Abstract-Conference.html},
  timestamp    = {Thu, 13 Feb 2025 16:56:43 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/PengS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
