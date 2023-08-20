r"""Shared modules used """
from torch import nn

class EDNCustomRNN(nn.Module):
    """Ref: """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        """
        recurrent, _ = self.rnn(input)  
        output = self.linear(recurrent)  
        return output
