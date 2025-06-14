import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW, get_scheduler
from PIL import Image
from tqdm.auto import tqdm

class ImageCaptioningModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Инициализация процессора и модели
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def train(self, processed_dataset, batch_size, lr, epochs, data_collator, path):
        train_dataset = processed_dataset['train']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        # Выбираем оптимизатор и планировщик (AdamW + Linear)
        # Можем увеличить num_warmup_steps, чтобы избежать резких скачков градиентов
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Цикл обучения
        progress_bar = tqdm(range(num_training_steps))
        self.model.train()

        for epoch in range(epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['input_ids']
                )

                loss = outputs.loss
                loss.backward()

                # Обновляем веса
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        # Сохраняем модель после обучения (Можем сделать чекпоинты после каждой эпохи в случае ошибки на N эпохе)
        self.save_model(path)

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def generate_caption(self, image_path, max_tokens):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
