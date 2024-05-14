import os
import torch
from prompt import fill_in_query_discrimination_template, fill_in_brief_nli_discrimination_template, fill_in_nli_discrimination_template

def score_relevance(character, statement, query, discriminator):
    
    if type(discriminator) != str:

        prompt = fill_in_query_discrimination_template(character, statement, query)

        scores = discriminator.classifier(**discriminator.tok(prompt, return_tensors="pt").to("cuda:0")).logits[0].softmax(-1)
    
    else:

        prompt = fill_in_query_discrimination_template(character, statement, query)
        
        system_prompt = "You are a helpful agent to build AI characters, your job is to determine whether an utterance from the human user to a role-playing AI should be responded by including the information in the given persona statement or not."
    
        response = openai.ChatCompletion.create(
        model=discriminator,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        ).choices[0]['message']["content"]
        
        scores = torch.FloatTensor([response.lower() == label for label in ["no", "yes"]])

    return scores

def score_nli(character, statement, query, response, discriminator):
    
    if type(discriminator) != str:
        
        prompt = fill_in_brief_nli_discrimination_template(character, statement, query, response)
        scores = discriminator.classifier(**discriminator.tok(prompt, return_tensors="pt").to("cuda:0")).logits[0].softmax(-1)
    
    else:
        
        system_prompt = "You are a helpful agent to build AI characters, your job is to discriminate whether the given persona statement is entailed, neutral, contradict to the response in natural language inference."
    
        prompt = fill_in_nli_discrimination_template(character, statement, query, response)
    
        response = openai.ChatCompletion.create(
        model=discriminator,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        ).choices[0]['message']["content"]
        
        scores = torch.FloatTensor([response.lower() == label for label in ["contradict", "neutral", "entailed"]])
    
    return scores

def score_apc(character, statement, query, response, relevance_discriminator, nli_discriminator):

    relevance_score = score_relevance(character, statement, query, relevance_discriminator)
    nli_score = score_nli(character, statement, query, response, nli_discriminator)

    apc_score = relevance_score[0] * (1 - nli_score[0]) + relevance_score[1] * nli_score[2]

    return apc_score

def score_APC(character, statements, query, response, relevance_discriminator, nli_discriminator):

    return torch.stack([score_apc(character, statement, query, response, relevance_discriminator, nli_discriminator) for statement in statements]).sum()