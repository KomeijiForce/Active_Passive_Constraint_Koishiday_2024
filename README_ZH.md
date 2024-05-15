# 恋之日2024： 人设主导的角色扮演任务的全局准确性的量化和优化

- 让幻想照进现实

## 引言 [\[论文\]](https://arxiv.org/abs/2405.07726)

以人设为驱动的角色扮演（Persona-driven Role-playing, PRP）允许你仅用几段简短的文字来描述一个AI角色的人设！然而，让AI角色忠实地遵守所有人设陈述是非常困难的。AI角色总是犯很多错误，或者在他们应该知道的知识上总是模棱两可。

![Case](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/case_koishi_zh.png)

这其中的主要原因是缺乏一个可以量化全球PRP忠诚度的指标。因此，我决定按照人类直觉来设计忠诚度的衡量指标:

![APC](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_koishi_zh.png)

简单来说，每当用户输入语句时，每个人设陈述将成为一个主动的（与用户语句相关）或被动的（与用户语句无关）约束。为了满足主动约束，回应需要由该人设所蕴含（包含该人设中的信息）。否则，对于被动约束，回应只需不与它们相矛盾（不包含与人设相违的信息）。

![DPO](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/dpo_koishi_zh.png)

我们将遍历所有人设陈述，并检查它们的约束是否得到满足。我们将计数满足的约束数量，用作评估全局PRP忠诚度的指标。这个指标被称为主动-被动约束（Active-Passive-Constraint, APC）得分。直接偏好优化（Direct Preference Optimization, DPO）是一种可以鼓励模型更符合人类或标准偏好的回应的方法。因此，我们可以针对同一用户语句抽样两个回应，然后根据它们的APC得分应用DPO，以鼓励PRP模型地更忠诚于全局人设陈述。

在实践中，APC得分由概率模型分配，以更准确地估计陈述与用户语句的相关性概率和陈述到回应的自然语言推理概率，具体如下，

![Formula](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_formula.png)

如果你不喜欢公式，你只需要知道我们需要两个概率估计器来评估相关性和自然语言推理（NLI）。

![Distillation](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/distillation_koishi_zh.png)

因此，我们使用上述流程通过从GPT-4提炼和合成数据集来构建这样的估计器。到目前为止，全球PRP忠诚度量化和优化的难题已经解决，让我们开始我们构建忠诚PRP代理的旅程，be of good cheer!

## 准备

在你开始旅程之前，你需要准备以下事项：

根据这个[页面](https://docs.anaconda.com/free/miniconda/miniconda-install/)的指示下载MiniConda3。

根据这个[页面](https://openai.com/index/openai-api/)的指示创建一个OpenAI账户并获取OpenAI API。

根据这个[页面](https://huggingface.co/settings/tokens)的指示创建一个Huggingface账户并创建一个Huggingface用于阅读的Token。

我的实现基于Gemma-1.1-7b-it，所以你需要根据这个[页面](https://huggingface.co/google/gemma-1.1-7b-it)的指示获取对Gemma模型的访问权限。

然后你可以创建一个环境并安装所需的Python包：

```bash
conda create -n apc python=3.8
conda activate apc
python -m pip install -r requirements.txt
```

## 快速上手

### 用基于APC的DPO训练RAG模型

我已将最忠诚的人设驱动角色扮演代理的学习场景形式化为一个简单的bash命令。你只需将 ```bash_is_all_you_need.sh```中的```openai_key``` 和 ```hf_token```替换为你自己的，然后运行
```bash
bash bash_is_all_you_need.sh
```

此脚本默认为Alice（详见```wiki```）构建了一个基于APC的DPO PRP系统，该系统使用RAG。你可以在```prp_models```中找到PRP代理的LoRA权重，在```statement```中找到中间数据集，在```discriminators```中找到中间鉴别器（如果你将```use_pretrained_discriminator```设置为False）。

你可以通过简单地将wiki文本（段落之间用"\n\n"分隔）放在```wiki```文件夹中，并命名为{character_name}_wiki.txt，为任何你喜欢的角色构建这个高级PRP系统。然后替换```bash_is_all_you_need.sh```中的```character```并运行它。你将在相应的目录中找到所需的一切。

我们已经优化了GPU的利用。然而，你仍然需要一个大于40G的GPU来运行这个bash命令。

- 参数建议

```model_engine```: "gpt-4"，提示是专为GPT-4编写的，使用其他LLMs可能会引起错误。

```use_pretrained_discriminator```: True，通常启用以减少生成相关性和NLI数据集的成本。（你仍然需要生成人物声明和用户查询！）

```prp_scale```: "7b", "2b" Gemma模型总是拒绝扮演角色。

```max_dpo_data```: 100，通常在一个小时内为具有大约100个人物声明的角色构建DPO数据集。

```lora_rank```: >= 32，较低的LoRA rank会损害角色扮演性能。

```rag_top_k```: 4-6，分析显示这一范围表现最佳。

使用APC得分评估响应
我们在```score.py```中基于```classifier.py```中定义的鉴别器实现了APC评分函数。使用```score_APC```函数，你可以根据所有人物声明评分不同响应的预期约束满足数，我们在```evaluation_example.py```中提供了一个使用案例，如下所示。
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

基于输出分数，你可以大致了解APC得分如何看待PRP的忠诚度。

### 与AI角色聊天！

在运行基于APC的DPO之后，你将在```prp_models/gemma-1.1-7b-it-lora-{character}-rag-dpo```获得你角色的LoRA权重，这可以用来与你的AI角色聊天。我们在```chat_example.py```中提供了一个示例，也如下所示。

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

以下是一个和古明地恋对话的示例:

```
User: Hi, Koishi! What is your ability?
Komeiji Koishi: I call it the "Silent Whisperer." It allows me to manipulate the unconsciousness of others, making me invisible and granting me control over their actions.
User: Where do you live?
Komeiji Koishi: The Palace of the Earth Spirits serves as my humble abode.
User: Who is your sister?
Komeiji Koishi: Satori Komeiji. The one with all the serious face. 😜
```

目前，由于我们论文中讨论的主题范围，系统仅支持单轮对话。我们将很快投入更多的工程努力以支持多轮对话！

## 数据集和模型

合成的人设-用户语句相关性数据集: [KomeijiForce/role-playing-apc-relevance](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-relevance) (英文), [KomeijiForce/role-playing-apc-multilingual-relevance](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-multilingual-relevance) (多语言)

合成的人设-回应自然语言推理数据集: [KomeijiForce/role-playing-apc-nli](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-nli) (英文), [KomeijiForce/role-playing-apc-multilingual-nli](https://huggingface.co/datasets/KomeijiForce/role-playing-apc-multilingual-nli) (多语言)

人设-用户语句相关性分类器: [KomeijiForce/deberta-v3-large-relevance-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-relevance-12character) (英文), [KomeijiForce/xlm-roberta-large-relevance-multilingual-12character](https://huggingface.co/datasets/KomeijiForce/xlm-roberta-large-relevance-multilingual-12character) (多语言)

人设-回应自然语言推理分类器: [KomeijiForce/deberta-v3-large-relevance-12character](https://huggingface.co/KomeijiForce/deberta-v3-large-nli-12character) (英文), [KomeijiForce/xlm-roberta-large-nli-multilingual-12character](https://huggingface.co/KomeijiForce/xlm-roberta-large-nli-multilingual-12character) (多语言)
### PRP数据集统计性质

| 角色 | 人设陈述 | 采访问题 | 相关性数据 | 推理数据 |
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

### 性能

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




## 待解决事项

- 支持更多语言
- 支持多轮对话
- 引入更多个性化训练参数

## 引用

@article{apc4prp,
  title={Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing},
  author={Peng, Letian and Shang, Jingbo},
  journal={arXiv preprint arXiv:2405.07726},
  year={2024}
}
