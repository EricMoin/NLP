import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

from config import Config


class BERTNERModel(nn.Module):
    model: AutoModel
    tokenizer: AutoTokenizer

    def __init__(self, config: Config, model_name="bert-base-chinese"):
        super(BERTNERModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.model.config.hidden_size, len(config.tag2idx))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 只计算active部分的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

        return loss, logits
