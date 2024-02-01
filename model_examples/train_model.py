"""
PRE_REQUISITE TO RUN THIS EXAMPLE
Requirements to be installed before running the custom model
    !pip install torch==2.1.0
    !pip install torchdata==0.7.0
    !pip install torchtext==0.16.0
    !pip install portalocker==2.7.0


"""
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The code below is your Train
class DatasetHandler:
    def __init__(
        self,
        training_data_iterator,
        testing_data_iterator,
        batch_size: int,
        train_split_percent: float = 0.95,
    ):
        self._tokenizer = get_tokenizer("basic_english")
        self._vocab = self._build_vocabulary(training_data_iterator)
        self._train_dataset = to_map_style_dataset(training_data_iterator)
        self._test_dataset = to_map_style_dataset(testing_data_iterator)

        num_train = int(len(self._train_dataset) * train_split_percent)
        self._train_dataset, self._validate_dataset = random_split(
            self._train_dataset, [num_train, len(self._train_dataset) - num_train]
        )
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def _build_vocabulary(self, training_data_iterator):
        def tokenize(data_iterator):
            for _, text in data_iterator:
                yield self._tokenizer(text)

        vocab = build_vocab_from_iterator(
            tokenize(training_data_iterator), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])

        return vocab

    def _collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(self.process_label(_label))
            processed_text = torch.tensor(self.process_text(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        return label_list.to(device), text_list.to(device), offsets.to(device)

    def process_text(self, text: str):
        return self._vocab(self._tokenizer(text))

    def process_label(self, label: int):
        return int(label) - 1

    def get_train_dataset_loader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )

    def get_test_dataset_loader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )

    def get_validate_dataset_loader(self):
        return DataLoader(
            self._validate_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )

    def num_classes(self):
        return len(set([label for (label, text) in self._train_dataset.dataset._data]))

    def vocab_size(self):
        return len(self._vocab)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets=None):
        if offsets is None:
            offsets = torch.tensor([0])
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


@dataclass
class ModelTrainingSettings:
    epochs: int
    learning_rate: int
    loss_criterion: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_handler: DatasetHandler,
        settings: ModelTrainingSettings,
    ):
        self._training_settings = settings
        self._model = model
        self._dataset_handler = dataset_handler
        self._training_dataset = self._dataset_handler.get_train_dataset_loader()
        self._testing_dataset = self._dataset_handler.get_test_dataset_loader()
        self._validation_dataset = self._dataset_handler.get_validate_dataset_loader()
        self._optimizer = self._training_settings.optimizer
        self._lr_scheduler = self._training_settings.lr_scheduler
        self._loss_criterion = self._training_settings.loss_criterion

    @property
    def model(self):
        return self._model

    @staticmethod
    def _print_metrics(epoch, idx, batches, accuracy):
        print(
            "| epoch {:3d} | {:5d}/{:5d} batches "
            "| accuracy {:8.3f}".format(epoch, idx, batches, accuracy)
        )

    @staticmethod
    def _print_epoch(epoch, epoch_start_time, validation_accuracy):
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, validation_accuracy
            )
        )
        print("-" * 59)

    def _training_iteration(self, epoch: int):
        self._model.train()

        total_correct, total_count = 0, 0
        log_interval = 500

        for idx, (label, text, offsets) in enumerate(self._training_dataset):
            self._optimizer.zero_grad()
            predicted_label = self._model(text, offsets)
            loss = self._loss_criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
            self._optimizer.step()
            total_correct += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                accuracy = total_correct / total_count
                ModelTrainer._print_metrics(
                    epoch, idx, len(self._training_dataset), accuracy
                )

    def evaluate_accuracy(self, dataloader):
        self._model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self._model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def train(self):
        total_accu = None
        for epoch in range(1, self._training_settings.epochs + 1):
            epoch_start_time = time.time()
            self._training_iteration(epoch=epoch)
            accu_val = self.evaluate_accuracy(self._validation_dataset)
            if total_accu is not None and total_accu > accu_val:
                self._lr_scheduler.step()
            else:
                total_accu = accu_val
            ModelTrainer._print_epoch(epoch, epoch_start_time, accu_val)

    def predict(self, text):
        with torch.no_grad():
            # preprocess text
            text = torch.tensor(self._dataset_handler.process_text(text))
            # predict using pytorch model
            output = self._model(text, torch.tensor([0]))
            # get prediction label
            prediction = output.argmax(1).item()
            # convert integer label to meaningful label
            ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
            # return meaningful label
            return ag_news_label[prediction]


train_iter, test_iter = AG_NEWS()
dataset_handler = DatasetHandler(train_iter, test_iter, batch_size=64)
embedding_size = 64

model = TextClassificationModel(
    dataset_handler.vocab_size(), embedding_size, dataset_handler.num_classes()
)

max_epochs = 2
lr = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model_training_settings = ModelTrainingSettings(
    epochs=max_epochs,
    learning_rate=5,
    loss_criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=scheduler,
)


def get_trained_model():
    model_trainer = ModelTrainer(
        model=model, dataset_handler=dataset_handler, settings=model_training_settings
    )

    model_trainer.train()
    return model_trainer.model
