from datasets import load_dataset
from PIL import Image

class DatasetLoader:
    def __init__(self, processor):
        self.processor = processor

    def load_dataset(self, dataset_path):
        return load_dataset('json', data_files=dataset_path)

    def preprocess_function(self, examples):
        images = []
        for image_file in examples['image']:
            image = Image.open(image_file).convert('RGB')
            images.append(image)
        inputs = self.processor(
            images=images,
            text=examples['caption'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        # Возвращаем только нужные ключи
        inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values']
        }
        return inputs

