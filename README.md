# Koishiday's 2024: Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing

恋之日2024： 人设主导的角色扮演任务的全局准确性的量化和优化

こいしの日2024： ペルソナ主導のロールプレイングにおける全体的忠実度の定量化および最適化

- Let there be fantasy

- 让幻想照进现实

- 幻想を現に

# Introduction

Persona-driven Role-playing (PRP) is so cool that it allows you to build AI characters with several short paragraphs to describe the persona (人设/設定)! However, how to keep the AI character faithfully to **ALL** persona statements is a hard problem. PRP agents always make bunches of mistakes or are always vague about the knowledge they should know. 

![APC](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/case_koishi.png)

The main reason behind this limitation is the lack of a metric that can quantify the global PRP faithfulness. So I decide to do so following this intuition:

![APC](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_koishi.png)



# Preparation

Before your journey, you need to prepare the following stuff:

1. Download MiniConda3 following the instructions on this [page](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Create an OpenAI account and get an OpenAI API following the instructions on this [page](https://openai.com/index/openai-api/).

3. Create a Huggingface account and create a Huggingface Token for reading following the instructions on this [page](https://huggingface.co/settings/tokens).

4. Gain access to Gemma models following the instructions on this [page](https://huggingface.co/google/gemma-1.1-7b-it).

Then you can create an environment and install the required Python packages:

```bash
conda create -n apc python=3.8
conda activate apc
python -m pip install -r requirements.txt
```

# Quick Start

I have formalized the learning scenario for the most faithful persona-driven role-playing agent as a simple bash command. You only have to replace the ```openai_key``` and ```hf_token``` in ```bash_is_all_you_need.sh``` with your own, and then run
```bash
bash bash_is_all_you_need.sh
```

This script builds an APC-based DPO PRP system with RAG for Alice (detailed in ```wiki```) by default. You can find the LoRA weights of the PRP agent in ```prp_models```, the intermediate datasets in ```statement```, and intermediate discriminators in ```discriminators```.

You can build this advanced PRP system for any character you like by simply putting a wiki in ```wiki``` with name ```{character_name}_wiki.txt```. Then replace the ```character``` in the ```bash_is_all_you_need.sh``` and run it. You will find everything you need in the corresponding directories.

We have optimized the GPU usage for implementation. However, you still need a >32G GPU to run the bash command.
