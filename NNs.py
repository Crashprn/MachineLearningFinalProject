from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = activation


        self.stem = nn.Sequential(nn.Linear(input_size, hidden_size), activation)

        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation)
        
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x += layer(x)
        x = self.output(x)
        return x
