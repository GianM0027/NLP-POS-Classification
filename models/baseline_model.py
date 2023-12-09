from drTorch.modules import TrainableModule
import torch
from torchtext.vocab import GloVe

import pandas as pd


def create_embedding_matrix(vocabulary:pd.Series,
                            embedding_dim:int,
                            glove_model_version:str,
                            high:float,
                            low:float) -> torch.Tensor:
    """
    Create an embedding matrix for a given vocabulary using GloVe word embeddings.

    :param vocabulary: A pandas Series containing the vocabulary words.
    :param embedding_dim: The dimension of the word embeddings.
    :param glove_model_version: The version of the GloVe pre-trained model to use.
                                possible values: "42B", "840B" "twitter.27B" "6B"
    :param high: The upper bound for initializing out-of-vocabulary (OOV) embeddings.
    :param low: The lower bound for initializing out-of-vocabulary (OOV) embeddings.
    :return: A torch.Tensor representing the embedding matrix.
    """

    oov_embedding = (high - low) * torch.rand(embedding_dim) + low

    glove_embedder = GloVe(name=glove_model_version, dim=embedding_dim)
    vocabulary_embedding = {}
    index = 0
    for word in vocabulary:
        if word in glove_embedder:
            vocabulary_embedding[word] = index, glove_embedder[word]
        else:
            vocabulary_embedding[word] = index, (high - low) * torch.rand(embedding_dim) + low
        index += 1

    words = vocabulary_embedding.keys()
    embedding_matrix = torch.zeros(len(words) + 2, embedding_dim, dtype=torch.float)

    for key in words:
        idx = vocabulary_embedding[key][0]
        embedding_matrix[idx] = vocabulary_embedding[key][1]

    embedding_matrix[-2] = torch.zeros(embedding_dim, dtype=torch.float)     # Padding
    embedding_matrix[-1] = oov_embedding                                     # Out_Of_Vocabulary
    return embedding_matrix


class BaselineModel(TrainableModule):

    def __init__(self,
                 vocabulary: pd.Series,
                 embedding_dim: int,
                 glove_model_version: str,
                 high: float,
                 low: float,
                 freeze: bool,
                 padding_index: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int):
        """
       Initialize a baseline model using an embedding layer, Bidirectional LSTM, and a fully connected layer.

       :param vocabulary: A pandas.Series containing the vocabulary words.
       :param embedding_dim: The dimension of the word embeddings.
       :param glove_model_version: The version of the GloVe pre-trained model to use.
       :param high: The upper bound for initializing out-of-vocabulary (OOV) embeddings.
       :param low: The lower bound for initializing out-of-vocabulary (OOV) embeddings.
       :param padding_index: The index for padding in the embedding matrix.
       :param hidden_size: The number of features in the hidden state of the LSTM.
       :param output_size: The size of the output layer.
       :param num_layers: The number of LSTM layers.
        """

        super(BaselineModel, self).__init__()

        embedding_matrix = create_embedding_matrix(vocabulary=vocabulary,
                                                   embedding_dim=embedding_dim,
                                                   glove_model_version=glove_model_version,
                                                   high=high,
                                                   low=low)

        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_index)

        self.lstm = torch.nn.LSTM(embedding_dim,
                                  hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=True)

        self.fc = torch.nn.Linear(2 * hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters:
        :param x: Input data.

        :returns: output: Model's output.
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)

        return output

    def get_embedding_weights_clone(self) -> torch.Tensor:
        """
        Get the weights of the embedding layer.

        Returns:
            torch.Tensor: A tensor containing the weights of the embedding layer.

        Note:
            The returned tensor is a reference to the internal weights of the embedding layer.
            Modifying this tensor will affect the embedding layer. Pay attention.
        """
        return self.embedding.weight.clone()
