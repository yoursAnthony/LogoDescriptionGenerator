import torch

# Объединяем элементы датасета в один батч
def data_collator(features):
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'].squeeze() for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'].squeeze() for f in features])
    batch['pixel_values'] = torch.stack([f['pixel_values'].squeeze() for f in features])
    batch['labels'] = batch['input_ids'].clone()
    return batch
