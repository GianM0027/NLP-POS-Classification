from .baseline_model import BaselineModel
import torch

import pandas as pd


class Model2(BaselineModel):

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
       Initialize a Model 2 using an embedding layer,  1 Bidirectional LSTM, and 2 fully connected layer.

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
        super(Model2, self).__init__(vocabulary=vocabulary,
                                     embedding_dim=embedding_dim,
                                     glove_model_version=glove_model_version,
                                     high=high,
                                     low=low,
                                     freeze=freeze,
                                     padding_index=padding_index,
                                     hidden_size=hidden_size,
                                     output_size=output_size,
                                     num_layers=num_layers,
                                     )

        self.fc2 = torch.nn.Linear(output_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        :param x: Input data.
        :returns: output (torch.Tensor): Model's output.
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out)
        x = self.relu(x)
        output = self.fc2(x)
        return output

