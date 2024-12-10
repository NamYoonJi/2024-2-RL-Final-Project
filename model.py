import torch
import torch.nn as nn

class PreLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(PreLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim) #각 차원의 평균과 표준편

    def forward(self, x):
        """
        Extract Feature of stock data
        Use hidden cell for the extracted feature
        """
        # Pass through LSTM
        _, (hn,_) = self.lstm(x)
        # Use the last hidden state (corresponding to the final time step)
        final_hidden_state = hn.squeeze()
        #print("hidden:", final_hidden_state.shape)
        

        # Pass through the linear layer to get the output features
        x = self.fc1(final_hidden_state)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

class PolicyNetwork(nn.Module):
    def __init__(self, pre_lstm, action_dim):
        super(PolicyNetwork, self).__init__()
        self.hidden_dim = 128
        self.pre_lstm = pre_lstm
        self.lstm = nn.LSTM(pre_lstm.fc3.out_features, self.hidden_dim, num_layers = 1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2*action_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the PolicyNetwork.

        Returns:
            torch.Tensor: Expectation vector and diagonal entries of covariance matrix for Multivariate Normal Dist.
        """
        features = self.pre_lstm(x)

        x = self.fc1(features)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        if x.ndim == 1:
          x = x.unsqueeze(0)
        mu = self.tanh(x[:,:3]).squeeze()
        sigma = self.sigmoid(x[:,3:]).squeeze()

        return mu, sigma

class ValueNetwork(nn.Module):
    def __init__(self, pre_lstm):
        super(ValueNetwork, self).__init__()
        self.hidden_dim = 128
        self.pre_lstm = pre_lstm
        self.lstm = nn.LSTM(pre_lstm.fc3.out_features, self.hidden_dim, num_layers = 1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        features = self.pre_lstm(x)
        x = self.fc1(features)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        output = self.fc3(x)

        return output