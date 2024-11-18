import os

class Config:
    #data_dir = os.path.join('data', 'ml-1m')
    model_save_dir = 'saved_models'
    
    # model hyperparameter
    max_len = 50
    hidden_units = 256
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    # training parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
    eval_step = 1000
    patience = 5
    k_values = [1, 5, 10]

config = Config()
