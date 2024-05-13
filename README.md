# Koishi's Day 2024: Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing

恋之日2024： 人设主导的角色扮演任务的全局准确性的量化和优化

こいしの日2024： ペルソナ主導のロールプレイングにおける全体的忠実度の定量化および最適化

- Let there be fantasy

- 让幻想照进现实

- 幻想を現に

## Introduction

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

This script builds an APC-based DPO PRP system with RAG for Alice (detailed in ```wiki```) by default. You can find the LoRA weights of the PRP agent in ```prp_models```, the intermediate datasets in ```statement```, and intermediate discriminators in ```discriminators```.

You can build this advanced PRP system for any character you like by simply putting a wiki text (paragraphs separated by "\n\n") in ```wiki``` with name ```{character_name}_wiki.txt```. Then replace the ```character``` in the ```bash_is_all_you_need.sh``` and run it. You will find everything you need in the corresponding directories.

We have optimized the GPU usage for implementation. However, you still need a >32G GPU to run the bash command.

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

Based on the output scores, you can have a rough understand of how APC score views PRP faithfulness.

## Datasets and Checkpoints

The synthesized dataset for statement-query relevance: [KomeijiForce/role-playing-apc-relevance](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-relevance)

The synthesized dataset for statement-to-response NLI: [KomeijiForce/role-playing-apc-nli](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-nli)

The fine-tuned DeBERTa-V3-Large discriminator for statement-query relevance: [KomeijiForce/deberta-v3-large-relevance-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-relevance-12character)

The fine-tuned DeBERTa-V3-Large discriminator for statement-to-response NLI: [KomeijiForce/deberta-v3-large-nli-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-nli-12character)

## Todo List

- Support More Languages
- Support Multi-turn Conversations
- Allow more Customized Training Setups
