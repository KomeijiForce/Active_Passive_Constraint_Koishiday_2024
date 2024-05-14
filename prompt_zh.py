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

    icl_character = "贝多芬"
    icl_passage = '''路德维希·范·贝多芬（德语：Ludwig van Beethoven；1770年12月16日—1827年3月26日），德意志作曲家、钢琴演奏家。贝多芬上承古典乐派传统，下启浪漫乐派之风格与精神，因而在音乐史上占有重要的地位。[1]贝多芬虽然经历听力下降，仍一直创作音乐，他一生创作了9部交响曲、36首钢琴奏鸣曲（其中32首带有编号，1首未完成，编号为WoO. 51）、10部小提琴奏鸣曲、16首弦乐四重奏、1部歌剧及2部弥撒曲等等。这些作品无论是在曲式、和声以及织体上都有重大创新，影响了音乐发展史，其中32首钢琴奏鸣曲，第三、五、六、九交响曲和《庄严弥撒》是其代表作，尤其闻名。1827年贝多芬因病逝世。在汉字文化圈，贝多芬有“乐圣”之尊称[2][3]。'''
    icl_output = '''- 贝多芬是一位德意志作曲家和钢琴演奏家，代表着音乐历史上从古典到浪漫的过渡。
- 贝多芬在经历听力丧失后，仍持续创作出众多影响深远的音乐作品。
- 贝多芬的作品包括9部交响曲、36首钢琴奏鸣曲、10部小提琴奏鸣曲以及其他重要作品，展现了曲式和和声上的重大创新。
- 贝多芬的第三、五、六和九交响曲以及《庄严弥撒》等作品尤为著名，体现了他在音乐上的杰出成就。
- 贝多芬因病逝世于1827年，但他的音乐遗产至今仍对世界音乐产生深远影响。
- 在汉字文化圈中，贝多芬被尊称为“乐圣”，体现了他在音乐史上的崇高地位。'''

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

            bar.set_description(f"正在生成相关的提问... 提问数量: {n_query}")
            
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
                
        bar.set_description(f"正在生成提问相关性分类数据集... 数据集大小： {len(new_dataset)}")
        
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

                bar.set_description(f"正在生成人设对回应的自然语言推理分类数据集... 数据集大小： {n_nli}")
            
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

        bar.set_description(f"正在生成人设对回应的自然语言推理分类数据集V2... 数据集大小： {len(new_dataset)}")
            
    json.dump(new_dataset, open(f"statement/{character}.nli.v2.json", "w"))
    
    return new_dataset