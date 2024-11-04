from typing import Dict

DEFAULT_CONFIG = {
    'model': {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
    }
}

def get_config() -> Dict:
    return DEFAULT_CONFIG.copy()
