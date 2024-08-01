from dataclasses import dataclass


##### Base Model #####
@dataclass
class BASE_CONFIG:
    """add Large model config"""
    model_link = ""

    input_size = (3, 224, 224)
    embedding_dim = 768


##### Large Model #####
@dataclass
class LARGE_CONFIG:
    """add Large model config"""
    model_link = ""

    input_size = (3, 224, 224)
    embedding_dim = 1024
