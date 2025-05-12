import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Definir o modelo LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=2, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Carregar os dados
with open('real_data_sequences.json', 'r') as f:
    data = json.load(f)

# Preparar os dados para treino
sequence_length = 5
inputs = []
targets = []
for sequence in data:
    sequence_inputs = []
    for frame in sequence:
        sequence_inputs.append([
            frame['y'], frame['psi'], frame['v'],
            frame['y_ref'], frame['psi_ref'], frame['v_ref']
        ])
    inputs.append(sequence_inputs)
    targets.append([sequence[-1]['delta'], sequence[-1]['a']])

inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Dividir em treino e validação
train_size = int(0.8 * len(inputs))
train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
train_targets, val_targets = targets[:train_size], targets[train_size:]

# Inicializar o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=6, hidden_size=64, output_size=2, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treino
num_epochs = 200
batch_size = 64
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(train_inputs.size()[0])
    
    for i in range(0, train_inputs.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_inputs = train_inputs[indices].to(device)
        batch_targets = train_targets[indices].to(device)
        
        # Reiniciar estados ocultos para cada batch
        outputs, _ = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validação
    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(val_inputs.to(device))
        val_loss = criterion(val_outputs, val_targets.to(device))
    
    print(f"Época [{epoch+1}/{num_epochs}], Loss de Treino: {loss.item():.4f}, Loss de Validação: {val_loss.item():.4f}")

# Salvar o modelo treinado
torch.save(model.state_dict(), "lstm_model_real_data.pth")
print("Modelo treinado salvo em 'lstm_model_real_data.pth'")