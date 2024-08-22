from pathlib import Path

def get_config():
    return {
        "batch_size": 25,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "ru",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload" : None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "train_size" : 0.9
    }
    
def get_gpt_config():
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "num_blocks": 6,
        "d_model": 768,
        "model_folder": "weights",
        "model_basename": "gptmodel_",
        "preload" : None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/gptmodel",
        "category": "raw_review_Digital_Music",
        "train_size" : 0.2
    }


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])