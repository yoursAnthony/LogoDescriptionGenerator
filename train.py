from dataset import DatasetLoader
from model import ImageCaptioningModel
from utils import data_collator

def main():
    # Params
    pretrained_model_name = "Salesforce/blip-image-captioning-base"
    batch_size = 4
    learning_rate = 5e-5    # Низкое значение для аккуратной корректировки весов
    num_epochs = 4
    save_path = 'fine-tuned-blip'
    
    model = ImageCaptioningModel(pretrained_model_name)
    dataset_loader = DatasetLoader(model.processor)

    # Загрузка и преобразование датасета
    dataset = dataset_loader.load_dataset('dataset.json')
    processed_dataset = dataset.map(dataset_loader.preprocess_function, batched=True, remove_columns=['image', 'caption'])
    processed_dataset.set_format(type='torch')

    model.train(processed_dataset, batch_size, learning_rate, num_epochs, data_collator, save_path)

if __name__ == "__main__":
    main()
