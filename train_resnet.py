from dataclasses import dataclass


@dataclass
class C:
    DATASET_NAME = "CricketEC"
    NUM_CLASSES = 14
    NUM_FRAMES = 16
    BATCH_SIZE = 16
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 1
    LSTM_DROPOUT = 0.4
    FC_DROPOUT = 0.5
    TRAIN_SIZE = 0.8
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 3
    NUM_EPOCHS = 20
