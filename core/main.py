from config import get_configs
from trainer import Trainer


if __name__ == '__main__':
    configs = get_configs()
    trainer = Trainer(configs)
    trainer.train()
