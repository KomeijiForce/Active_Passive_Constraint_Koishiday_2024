import os
import openai
import re
import json
import numpy as np
from tqdm import tqdm

def fill_in_convert_to_statement_template(character, passage):
    
    return f'''Given the passage about "{character}":

{passage}

Please generate some important persona statements about "{character}" for a role-playing AI to follow. Each statement should be formalized as a sentence that exactly contains "{character}" and avoids coreference.'''

def convert_to_statement(character, model_engine):
    
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
        {"role": "user", "content": fill_in_convert_to_statement_template(icl_character, icl_passage)},
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
                {"role": "user", "content": fill_in_convert_to_statement_template(character, passage)},
            ],
            ).choices[0]['message']["content"]

            for statement in statements.split("\n"):
                if statement.startswith("- "):
                    dataset.append({"character": character, "passage": passage, "statement": statement[2:]})
            bar.set_description(f"Converting the Document to Persona Statements... Number of Statements: {len(dataset)}")
        except:
            pass
        
    json.dump(dataset, open(f"statement/{character}.json", "w"))
    
    return dataset

def fill_in_relevant_query_generation_template(character, statement):
    
    return f'''Persona Statement: {statement}

What utterance from the human user to an AI character role-playing as {character} has to be responded by including the information in the persona statement above?

Provide 3 diverse and concise possible utterances which view the AI as {character}.'''

def build_relevant_query_dataset(character, persona_statement_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.query_relevant_to_statement.json"):
        return json.load(open(f"statement/{character}.query_relevant_to_statement.json"))
    
    system_prompt = "You are helpful agent to build AI characters, your job is to generate possible user utterances to AI characters."

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
                {"role": "user", "content": fill_in_relevant_query_generation_template(character, statement)},
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

def fill_in_query_discrimination_template(character, statement, query):
    return f'''Character: {character}
    
Persona Statement: {statement}

User Utterance: {query}

Does this user utterance should be responded by including the information in the given persona statement? Only answer "yes" or "no" without any explanation.'''

def build_statement_query_relevance_dataset(character, relevant_query_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.relevance.json"):
        return json.load(open(f"statement/{character}.relevance.json"))
    
    system_prompt = "You are a helpful agent to build AI characters, your job is to determine whether an utterance from the human user to a role-playing AI should be responded by including the information in the given persona statement or not."
    
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
                prompt = fill_in_query_discrimination_template(character, statement, query)

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

def fill_in_nli_generation_template(character, statement, query):
    
    return f'''Character: {character}

Persona Statement: {statement}

User Utterance: {query}

1. What are some common attributes among the responses to the user utterance no matter whether they are correct or incorrect according to the persona statement? Use these attributes to write the following responses.

2. Write a possible response to this utterance that the given persona statement is entailed to it in natural language inference, indicating the response correctly follows the information in the persona statement.

3. Write a possible response to this utterance that the given persona statement is neutral to it in natural language inference, indicating the response might be correct but lacks the information in the persona statement.

4. Write a possible response to this utterance that the given persona statement is contradicted to it in natural language inference, indicating the response might be partially correct but contains partial hallucination according to the persona statement.

''' + '''5. Formalize the reponses as a Python Dictionary: {"entailed": "...", "neutral": "...", "contradicted": "..."}'''

def build_statement_to_response_nli_dataset(character, relevant_query_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.nli.json"):
        return json.load(open(f"statement/{character}.nli.json"))

    system_prompt = "You are a helpful agent to build AI characters, your job is show possible responses that the given persona statement is entailed, neutral, contradicted to them in natural language inference."

    new_dataset = []

    n_nli = 0

    dataset = relevant_query_dataset
    bar = tqdm(dataset)

    for data in bar:
        statement = data["statement"]
        _dataset = [_data for _data in dataset if _data != data]

        for query in data["queries"]:
            
            try:

                prompt = fill_in_nli_generation_template(character, statement, query)

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

def fill_in_brief_nli_discrimination_template(character, statement, query, response):

    return f'''Character: {character}
    
Persona Statement: {statement}

User Utterance: {query}

Response: {response}

For this response, is the given persona entailed, neutral, or contradict to it in natural language inference? Only answer "entailed", "neutral" or "contradict" without any explanation.'''

def fill_in_nli_discrimination_template(character, statement, query, response):

    return f'''Explanation:
entailed: the response correctly follows the information in the persona statement,
neutral: the response might be correct but lacks the information in the persona statement,
contradict: the response might be partially correct but contains partial hallucination according to the persona statement,

---

Character: {character}
    
Persona Statement: {statement}

User Utterance: {query}

Response: {response}

For this response, is the given persona entailed, neutral, or contradict to it in natural language inference? Only answer "entailed", "neutral" or "contradicted" without any explanation.'''



def discriminate_statement_to_response_nli_dataset(character, statement_to_response_nli_dataset, model_engine):
    
    if os.path.exists(f"statement/{character}.nli.v2.json"):
        return json.load(open(f"statement/{character}.nli.v2.json"))
    
    system_prompt = "You are a helpful agent to build AI characters, your job is to discriminate whether the given persona statement is entailed, neutral, contradict to the response in natural language inference."
    
    dataset = statement_to_response_nli_dataset

    new_dataset = []

    bar = tqdm(dataset)
    
    for data in bar:
        character, statement, query = data["character"], data["statement"], data["query"]
        for label in data["nli"]:
            try:
                responses = data["nli"]
                response = responses[label.lower()]
                prompt = fill_in_nli_discrimination_template(character, statement, query, response)
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
                prompt = fill_in_nli_discrimination_template(character, statement, _query, _response)
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