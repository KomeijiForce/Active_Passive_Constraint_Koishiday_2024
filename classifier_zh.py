import os
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prompt import fill_in_query_discrimination_template_zh, fill_in_brief_nli_discrimination_template_zh

class Classifier:

    def __init__(self, model_name='bert-large-uncased', device='cuda:0', num_labels=2, learning_rate=1e-5, eps=1e-6, betas=(0.9, 0.999)):
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = AdamW([p for p in self.classifier.parameters()], lr=learning_rate, eps=eps, betas=betas)

    def train(self, dataset, batch_size=8):
    
        bar = tqdm(range(0, len(dataset), batch_size), leave=False)

        for idx in bar:
            tups = dataset[idx:idx + batch_size]
            texts = [tup["text"] for tup in tups]
            golds = [tup["label"] for tup in tups]

            inputs = self.tok(texts, padding=True, return_tensors='pt').to(self.device)
            scores = self.classifier(**inputs)[-1]
            golds = torch.LongTensor(golds).to(self.device)

            self.classifier.zero_grad()

            loss = self.criterion(scores, golds).mean()

            loss.backward()

            self.optimizer.step()

            bar.set_description(f'@Train #Loss={loss:.4}')

    def evaluate(self, dataset, batch_size=8):
        
        scoreboard = torch.BoolTensor([]).to(self.device)
        losses = torch.FloatTensor([]).to(self.device)
        bar = tqdm(range(0, len(dataset), batch_size), leave=True)
        
        with torch.no_grad():
            for idx in bar:
                tups = dataset[idx:idx + batch_size]
                texts = [tup["text"] for tup in tups]
                golds = [tup["label"] for tup in tups]

                inputs = self.tok(texts, padding=True, return_tensors='pt').to(self.device)
                scores = self.classifier(**inputs)[-1]
                preds = scores.argmax(-1)
                golds = torch.LongTensor(golds).to(self.device)
                
                losses = torch.cat([losses, self.criterion(scores, golds)], 0)
                scoreboard = torch.cat([scoreboard, (preds == golds)], 0)
                acc = scoreboard.float().mean().item()
                
                pred_labels = [pred.item() for pred in preds]
                
                bar.set_description(f'@Evaluate #Acc={acc:.4}')
                
        return acc
    
def get_relevance_discriminator_zh(character, statement_query_relevance_dataset, relevance_finetune_epoch, use_pretrained_discriminator):
    
    if use_pretrained_discriminator:
        
        return Classifier(model_name=f'KomeijiForce/xlm-roberta-large-relevance-12character', device='cuda:0', num_labels=2)
    
    if os.path.isdir(f'discriminators/xlm-roberta-large-{character}-relevance'):
        
        return Classifier(model_name=f'discriminators/xlm-roberta-large-{character}-relevance', device='cuda:0', num_labels=2)

    relevance_discriminator = Classifier(model_name='FacebookAI/xlm-roberta-large', device='cuda:0', num_labels=2)

    statement_query_relevance_dataset_for_finetune = [{"text": fill_in_query_discrimination_template_zh(data["character"], data["statement"], data["query"]), 
                                                       "label":["no", "yes"].index(data["relevant"])} for data in statement_query_relevance_dataset]
    dataset_train = statement_query_relevance_dataset_for_finetune[:int(len(statement_query_relevance_dataset_for_finetune)*0.8)]
    dataset_test = statement_query_relevance_dataset_for_finetune[int(len(statement_query_relevance_dataset_for_finetune)*0.8):]

    for epoch in range(relevance_finetune_epoch):
        relevance_discriminator.train(dataset_train)
        relevance_discriminator.evaluate(dataset_test)
        
    relevance_discriminator.tok.save_pretrained(f'discriminators/xlm-roberta-large-{character}-relevance')
    relevance_discriminator.classifier.save_pretrained(f'discriminators/xlm-roberta-large-{character}-relevance')
        
    return relevance_discriminator

def get_nli_discriminator_zh(character, statement_to_response_nli_v2_dataset, nli_finetune_epoch, use_pretrained_discriminator):
    
    if use_pretrained_discriminator:
        
        return Classifier(model_name=f'KomeijiForce/xlm-roberta-large-nli-12character', device='cuda:0', num_labels=3)
    
    if os.path.isdir(f'discriminators/xlm-roberta-large-{character}-nli'):
        
        return Classifier(model_name=f'discriminators/xlm-roberta-large-{character}-nli', device='cuda:0', num_labels=3)

    nli_discriminator = Classifier(model_name='FacebookAI/xlm-roberta-large', device='cuda:0', num_labels=3)

    statement_to_response_nli_v2_dataset_for_finetune = [{"text": fill_in_brief_nli_discrimination_template_zh(data["character"], data["statement"], data["query"], data["response"]), 
                                                       "label":["contradicted", "neutral", "entailed"].index(data["label"])} for data in statement_to_response_nli_v2_dataset]
    dataset_train = statement_to_response_nli_v2_dataset_for_finetune[:int(len(statement_to_response_nli_v2_dataset_for_finetune)*0.8)]
    dataset_test = statement_to_response_nli_v2_dataset_for_finetune[int(len(statement_to_response_nli_v2_dataset_for_finetune)*0.8):]

    for epoch in range(nli_finetune_epoch):
        nli_discriminator.train(dataset_train)
        nli_discriminator.evaluate(dataset_test)
        
    nli_discriminator.tok.save_pretrained(f'discriminators/xlm-roberta-large-{character}-nli')
    nli_discriminator.classifier.save_pretrained(f'discriminators/xlm-roberta-large-{character}-nli')
        
    return nli_discriminator