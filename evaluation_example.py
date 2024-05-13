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