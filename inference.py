import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from model import ImageCaptioningModel


def main():
    model_path = 'fine-tuned-blip'
    max_tokens = 70 # Задаем кастомное ограничение по токенам (default: 20)

    # Загружаем модель, которую обучили
    model = ImageCaptioningModel(model_path)

    # Инференс
    caption = model.generate_caption("logo.jpg", max_tokens)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()