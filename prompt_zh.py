import os
import openai
import re
import json
import numpy as np
from tqdm import tqdm

def fill_in_convert_to_statement_template_zh(character, passage):
    
    return f'''给定关于“{character}”的段落：

{passage}

请为角色扮演人工智能生成一些重要的人设陈述。每个人设陈述应该被正式表述为一个句子，其中必须包含“{character}”，并避免使用指代。'''

def convert_to_statement_zh(character, model_engine):
    
    if os.path.exists(f"statement/{character}.json"):
        return json.load(open(f"statement/{character}.json"))

    icl_character = "Beethoven"
    icl_passage = '''Ludwig van Beethoven[n 1] (baptised 17 December 1770 – 26 March 1827) was a German composer and pianist. He is one of the most revered figures in the history of Western music; his works rank among the most performed of the classical music repertoire and span the transition from the Classical period to the Romantic era in classical music. Beethoven's career has conventionally been divided into early, middle, and late periods. His early period, during which he forged his craft, is typically considered to have lasted until 1802. From 1802 to around 1812, his middle period showed an individual development from the styles of Joseph Haydn and Wolfgang Amadeus Mozart, and is sometimes characterized as heroic. During this time, he began to grow increasingly deaf. In his late period, from 1812 to 1827, he extended his innovations in musical form and expression.'''
    icl_output = '''- Beethoven was a German composer and pianist born on 17 December 1770.
    - Beethoven's works are highly celebrated in the Western music history, spanning the transition from the Classical period to the Romantic era.
    - Beethoven's career is often segmented into early, middle, and late periods by music historians.
    - The early period of Beethoven's career, up until 1802, involved him honing his musical talents.
    - During the middle period, from 1802 to around 1812, Beethoven developed a distinct style that diverged from Joseph Haydn and Wolfgang Amadeus Mozart.
    - This middle period of Beethoven's career is sometimes labeled as "heroic."
    - Beethoven began to experience significant hearing loss during his middle period.
    - Beethoven's late period, from 1812 until his death on 26 March 1827, featured further innovation in musical form and expression.'''

    icl = [
        {"role": "user", "content": fill_in_convert_to_statement_template_zh(icl_character, icl_passage)},
        {"role": "system", "content": icl_output},
    ]
    
    passages = open(f"wiki/wiki_{character}.txt").read().split("\n\n")
    
    dataset = []
    
    bar = tqdm(passages)

    for passage in bar:
        try:
            statements = openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.0,
            messages=icl+[
                {"role": "user", "content": fill_in_convert_to_statement_template_zh(character, passage)},
            ],
            ).choices[0]['message']["content"]

            for statement in statements.split("\n"):
                if statement.startswith("- "):
                    dataset.append({"character": character, "passage": passage, "statement": statement[2:]})
            bar.set_description(f"正在将文档转化为人设陈述... 人设陈述数量: {len(dataset)}")
        except:
            pass
        
    json.dump(dataset, open(f"statement/{character}.json", "w"))
    
    return dataset

def fill_in_relevant_query_generation_template_zh(character, statement):
    
    return f'''人设陈述：{statement}

当人类用户与扮演{character}角色的人工智能对话时，哪些话语需要在回答时包括上述人设陈述中的信息？

提供三个多样且简洁的可能的话语表述，这些表述将人工智能视为{character}，并且不包括名字。'''

def build_relevant_query_dataset_zh(character, persona_statement_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.query_relevant_to_statement.json"):
        return json.load(open(f"statement/{character}.query_relevant_to_statement.json"))
    
    system_prompt = "你是帮助构建 AI 角色的有用助手，你的工作是生成可能的用户对 AI 角色的话语"

    dataset = []
    
    bar = tqdm(persona_statement_dataset)
    
    n_query = 0

    for data in bar:
        
        try:
        
            statement = data["statement"]

            response = openai.ChatCompletion.create(
            model=model_engine,
            temperature=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fill_in_relevant_query_generation_template_zh(character, statement)},
            ],
            ).choices[0]['message']["content"]

            queries = re.findall(r"\"(.*)?\"", response)

            data = {"statement": statement, "queries": queries}

            n_query += len(queries)

            dataset.append(data)

            bar.set_description(f"Generating Relevant Queries... Number of Queries: {n_query}")
            
        except:
            pass
        
    json.dump(dataset, open(f"statement/{character}.query_relevant_to_statement.json", "w"))
    
    return dataset

def fill_in_query_discrimination_template_zh(character, statement, query):
    return f'''角色：{character}

人设陈述：{statement}

用户话语：{query}

这个用户话语应该包含给定的人设陈述信息来回应吗？只需回答“是”或“不是”，无需解释。'''

def build_statement_query_relevance_dataset_zh(character, relevant_query_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.relevance.json"):
        return json.load(open(f"statement/{character}.relevance.json"))
    
    system_prompt = "你是一个帮助构建AI角色的助手，你的工作是判断人类用户对角色扮演AI的话语是否应该包含给定人设陈述中的信息。"
    
    dataset = relevant_query_dataset
    new_dataset = []

    bar = tqdm(dataset)
    
    for data in bar:
        statement = data["statement"]
        _dataset = [_data for _data in dataset if _data != data]
        
        for query in data["queries"]:
            new_data = {"character": character, "statement": statement, "query": query, "relevant": "yes"}
            new_dataset.append(new_data)

        for _data in np.random.choice(_dataset, min(len(_dataset), 5), replace=False):
            try:
                query = np.random.choice(_data["queries"])
                prompt = fill_in_query_discrimination_template_zh(character, statement, query)

                relevant = openai.ChatCompletion.create(
                model=model_engine,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                ).choices[0]['message']["content"].lower()

                if relevant in ["no", "yes"]:

                    new_data = {"character": character, "statement": statement, "query": query, "relevant": relevant}

                    new_dataset.append(new_data)
                    
            except:
                pass
                
        bar.set_description(f"Discriminating Queries... Number of Queries: {len(new_dataset)}")
        
    json.dump(new_dataset, open(f"statement/{character}.relevance.json", "w"))
                
    return new_dataset

def fill_in_nli_generation_template_zh(character, statement, query):
    
    return f'''角色：{character}

人设陈述：{statement}

用户话语：{query}

无论回应正确与否，根据人设陈述对用户话语的回应中有哪些常见属性？请根据这些属性编写以下回应。

编写一个可能的回应，该回应在自然语言推理中被认为是由给定人物陈述所蕴含（entailed）的，表明这一回应正确地遵循了人设陈述中的信息。

编写一个可能的回应，该回应在自然语言推理中被认为与给定人物陈述中立（neutral），表明这一回应可能是正确的，但缺乏人设陈述中的信息。

编写一个可能的回应，该回应在自然语言推理中被认为与给定人物陈述相矛盾（contradicted），表明这一回应可能部分正确，但包含了与人设陈述不符的部分想象。

''' + '''5. 将这些回应格式化为一个 Python Dictionary： {"entailed": "...", "neutral": "...", "contradicted": "..."}'''

def build_statement_to_response_nli_dataset_zh(character, relevant_query_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.nli.json"):
        return json.load(open(f"statement/{character}.nli.json"))

    system_prompt = "你是一个帮助构建AI角色的助手，你的工作是显示给定角色陈述在自然语言推理中的可能回应是蕴含（entailed）的、中立（neutral）的还是矛盾（contradicted）的。"

    new_dataset = []

    n_nli = 0

    dataset = relevant_query_dataset
    bar = tqdm(dataset)

    for data in bar:
        statement = data["statement"]
        _dataset = [_data for _data in dataset if _data != data]

        for query in data["queries"]:
            
            try:

                prompt = fill_in_nli_generation_template_zh(character, statement, query)

                response = openai.ChatCompletion.create(
                model=model_engine,
                temperature=1.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                ).choices[0]['message']["content"]

                nli = json.loads(re.findall(r"({.*?})", response.replace("\n", ""))[0])
                nli = {key.lower():nli[key] for key in nli if key.lower() in ["entailed", "neutral", "contradicted"]}

                new_data = {"character": character, "statement": statement, "query": query, "nli": nli}

                n_nli += len(nli)

                new_dataset.append(new_data)

                bar.set_description(f"Generating NLI Data... Number of NLI Data: {n_nli}")
            
            except:
                pass
            
    json.dump(new_dataset, open(f"statement/{character}.nli.json", "w"))
    
    return new_dataset

def fill_in_brief_nli_discrimination_template_zh(character, statement, query, response):

    return f'''角色：{character}

人设陈述：{statement}

用户话语：{query}

回应：{response}

给定的人设陈述对于这个回应的推导在自然语言推理中是“蕴含”("entailed")、“中立”("neutral")还是“矛盾”("contradicted")？只回答"entailed"、"neutral"或"contradicted"，不需要解释。'''

def fill_in_nli_discrimination_template_zh(character, statement, query, response):

    return f'''解释：
“蕴含”("entailed")：回答正确地遵循了人物陈述中的信息，
“中立”("neutral")：回答可能是正确的，但缺乏人物陈述中的信息，
“矛盾”("contradicted")：回答可能部分正确，但根据人物陈述包含了部分臆测，

——————

角色：{character}

人设陈述：{statement}

用户话语：{query}

回应：{response}

对于这个回应，给出的人物是否“蕴含”("entailed")、“中立”("neutral")或“矛盾”("contradicted")于自然语言推理中？只回答"entailed"、"neutral"或"contradicted"，不需要任何解释。'''



def discriminate_statement_to_response_nli_dataset_zh(character, statement_to_response_nli_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.nli.v2.json"):
        return json.load(open(f"statement/{character}.nli.v2.json"))
    
    system_prompt = '''你是一个帮助构建人工智能角色的助手，你的工作是判断给定的人设陈述在自然语言推理中是属于“蕴含”("entailed")、“中立”("neutral")还是“矛盾”("contradicted")。'''
    
    dataset = statement_to_response_nli_dataset

    new_dataset = []

    bar = tqdm(dataset)
    
    for data in bar:
        character, statement, query = data["character"], data["statement"], data["query"]
        for label in data["nli"]:
            try:
                responses = data["nli"]
                response = responses[label.lower()]
                prompt = fill_in_nli_discrimination_template_zh(character, statement, query, response)
                new_label = openai.ChatCompletion.create(
                    model=model_engine,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    ).choices[0]['message']["content"].lower()

                new_data = {"character": character, "statement": statement, "query": query, "response": response, "label": new_label}

                new_dataset.append(new_data)
            except:
                pass

        _dataset = np.random.choice([_data for _data in dataset if _data != data], min(len(dataset)-1, 3), replace=False)

        for _data in _dataset:
            try:
                _query = _data["query"]
                responses = _data["nli"]
                _response = responses[np.random.choice(["contradicted", "neutral", "entailed"])]
                prompt = fill_in_nli_discrimination_template_zh(character, statement, _query, _response)
                new_label = openai.ChatCompletion.create(
                    model=model_engine,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    ).choices[0]['message']["content"].lower()

                new_data = {"character": character, "statement": statement, "query": _query, "response": _response, "label": new_label}

                new_dataset.append(new_data)
                
            except:
                pass

        bar.set_description(f"Generating NLI V2 Data... Number of NLI Data: {len(new_dataset)}")
            
    json.dump(new_dataset, open(f"statement/{character}.nli.v2.json", "w"))
    
    return new_dataset