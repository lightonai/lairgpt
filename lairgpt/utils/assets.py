from enum import Enum
from os.path import expanduser
from lairgpt.utils.remote import local_dir

class Config(Enum):
    """Settings for preconfigured models instances
    """
    SMALL = {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "vocab_size": 50262,
        "max_seq_len": 1024
    }
    MEDIUM = {
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "vocab_size": 50262,
        "max_seq_len": 1024
    }
    LARGE = {
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "vocab_size": 50262,
        "max_seq_len": 1024
    }
    XLARGE = {
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "vocab_size": 50262,
        "max_seq_len": 1024
    }

class Snapshot(Enum):
    """Snapshots for preconfigured models state dictionaries
    """
    SMALL   = local_dir + "small.pt"
    MEDIUM  = local_dir + "medium.pt"
    LARGE   = local_dir + "large.pt"
    XLARGE  = local_dir + "xlarge.pt"

class Tokenizer(Enum):
    """Tokenizers for preconfigured models inference
    """
    CCNET = local_dir + "tokenizer_ccnet.json"
