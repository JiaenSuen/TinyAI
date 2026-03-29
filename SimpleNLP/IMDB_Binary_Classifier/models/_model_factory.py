from models.mlp import MLP
from models.rnn import EmbeddingLSTM
from models.gru import EmbeddingGRU
from models.textcnn import TextCNN


def build_model(model_name, vocab_size=None):
    if model_name == 'mlp':
        return MLP(input_dim=10000, hidden_dim=512, num_classes=2)

    elif model_name == 'lstm':
        return EmbeddingLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            num_classes=2
        )

    elif model_name == 'gru':
        return EmbeddingGRU(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            num_classes=2
        )

    elif model_name == 'textcnn':
        return TextCNN(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_classes=2,
            kernel_sizes=[3, 4, 5],
            num_filters=100
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")