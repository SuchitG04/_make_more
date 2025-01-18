import torch

from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

@dataclass
class TrainerConfig:
    lr: float = 1e-3
    batch_size: int = 32
    num_iters: int = 5000
    device: str = 'auto'

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataset: Dataset,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.optimizer = None

        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model.to(self.device)
        self.lossi = []


    @staticmethod
    def get_default_config():
        return TrainerConfig()


    def train(self):
        # !!! does not create copies !!!
        model, config = self.model, self.config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        # set model in train mode
        model.train()

        data_iter = iter(train_loader)
        # training loop
        for iter_num in range(config.num_iters):

            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]

            x, y = batch

            logits, loss = model(x, y)
            self.lossi.append(loss.item())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if iter_num % 100 == 0:
                print(f"iter {iter_num}; loss: {loss.item()}")

