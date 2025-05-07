import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

delta_rate_max = 0.1
vmax = 2.52

# Definição da RNN (usando LSTM)
class RNNForMPC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length):
        super(RNNForMPC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out, (hn, cn)

# Função para gerar dados de treinamento
def generate_training_data(num_sequences, sequence_length, dt=0.1, L=0.5):
    """
    Gera dados simulados para treinar a RNN, usando o modelo cinemático de bicicleta.
    Args:
        num_sequences: Número de sequências de treinamento
        sequence_length: Comprimento de cada sequência
        dt: Intervalo de tempo (segundos)
        L: Distância entre eixos (wheelbase, em metros)
    Returns:
        inputs: Tensor de shape (num_sequences, sequence_length, 6) com [x, y, ψ, v, δ, a]
        targets: Tensor de shape (num_sequences, sequence_length, 4) com [ẋ, ẏ, ψ̇, v̇]
    """
    inputs = []
    targets = []

    for _ in range(num_sequences):
        # Inicializar estado
        x = 0.0  # Posição longitudinal inicial (m)
        y = np.random.uniform(-0.5, 0.5)  # Posição transversal inicial (m)
        psi = np.random.uniform(-np.pi/4, np.pi/4)  # Yaw inicial (rad)
        v = np.random.uniform(0.5, 2.0)  # Velocidade inicial (m/s)

        # Simular uma trajetória ideal (curva suave)
        a_coeff = np.random.uniform(-0.01, 0.01)  # Curvatura
        b = np.random.uniform(-0.1, 0.1)  # Inclinação
        c = 0.0  # Centro
        def get_ideal_trajectory(x):
            return a_coeff * x**2 + b * x + c  # y = f(x)

        def get_psi_ref(x):
            dy_dx = 2 * a_coeff * x + b
            return np.arctan(dy_dx)  # ψ_ref = atan(dy/dx)

        sequence_inputs = []
        sequence_targets = []

        for t in range(sequence_length):
            # Calcular a posição e yaw de referência
            y_ref = get_ideal_trajectory(x)
            psi_ref = get_psi_ref(x)
            e_psi = psi - psi_ref  # Erro de heading

            # Gerar controles
            delta = np.random.uniform(-0.5, 0.5)  # Ângulo de direção (rad)
            if y_ref - y > 0:  # Desvio à direita
                delta += np.random.uniform(0.05, 0.2)
            elif y_ref - y < 0:  # Desvio à esquerda
                delta -= np.random.uniform(0.05, 0.2)
            delta = np.clip(delta, -0.5, 0.5)
            a = np.random.uniform(-1.0, 1.0)  # Aceleração (m/s²)

            # Calcular taxas de mudança (modelo cinemático)
            x_dot = v * np.cos(psi)  # m/s
            y_dot = v * np.sin(psi)  # m/s
            psi_dot = (v / L) * np.tan(delta)  # rad/s
            v_dot = a  # m/s²

            # Adicionar ruído realista
            x_dot += np.random.normal(0, 0.1)  # Ruído de 0.1 m/s
            y_dot += np.random.normal(0, 0.1)  # Ruído de 0.1 m/s
            psi_dot += np.random.normal(0, 0.005)  # Ruído de 0.005 rad/s
            v_dot += np.random.normal(0, 0.05)  # Ruído de 0.05 m/s²

            sequence_inputs.append([x, y, psi, v, delta, a])
            sequence_targets.append([x_dot, y_dot, psi_dot, v_dot])

            # Atualizar o estado
            x += x_dot * dt
            y += y_dot * dt
            psi += psi_dot * dt
            v += v_dot * dt
            v = np.clip(v, 0.5, 2.0)  # Limitar velocidade

            # Normalizar psi
            psi = np.arctan2(np.sin(psi), np.cos(psi))

        inputs.append(sequence_inputs)
        targets.append(sequence_targets)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return inputs, targets

# Função para treinar a RNN
def train_rnn(model, inputs, targets, num_epochs, batch_size, learning_rate, device):
    model = model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_sequences = inputs.shape[0]
    print(f"Começando o treinamento com {num_sequences} sequências...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        indices = torch.randperm(num_sequences)
        for i in range(0, num_sequences, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]

            optimizer.zero_grad()
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)

        avg_loss = total_loss / num_sequences
        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{num_epochs}], Perda média: {avg_loss:.4f}")

    print("Treinamento concluído!")

# Função para um passo do MPC com gradientes
def mpc_step(model, current_state, y_ref, psi_ref, horizon_length=5, dt=0.03, L=0.15, device="cpu"): #N x dt = Prediction Horizon : Tp
    """
    Executa um passo do MPC para otimizar os controles δ e a usando gradientes.
    Args:
        model: Modelo RNN treinado (LSTM)
        current_state: Estado atual [x, y, ψ, v]
        y_ref: Posição transversal desejada (m)
        psi_ref: Yaw desejado (rad)
        horizon_length: Comprimento do horizonte de predição
        dt: Intervalo de tempo (segundos)
        L: Distância entre eixos (m)
        device: Dispositivo para inferência ("cuda" ou "cpu")
    Returns:
        delta_opt, a_opt: Controles otimizados para o próximo timestep
    """
    model.eval()
    model = model.to(device)

    # Estado atual
    x, y, psi, v = current_state

    # Inicializar controles como tensores com gradientes
    delta_controls = torch.zeros(horizon_length, dtype=torch.float32, device=device, requires_grad=True)
    a_controls = torch.zeros(horizon_length, dtype=torch.float32, device=device, requires_grad=True)

    # Otimizador para os controles
    optimizer = optim.Adam([delta_controls, a_controls], lr=0.01)

    # Pesos para o custo
    w_y = 1.0  # Peso para erro lateral Qey
    w_psi = 0.5  # Peso para erro de heading Qyaw
    #w_v = v - vref # Peso de erro de velocidade Qv(v-vref)**2
    w_delta = 0.1  # Peso para suavidade do control Rdelta
    w_acc = 0.1  # Peso para suavidade dos control Racc

    # Número de iterações de otimização
    num_iterations = 50

    for _ in range(num_iterations):
        optimizer.zero_grad()

        # Inicializar estado para predição
        state = [x, y, psi, v]
        states = [state]
        h0 = None
        c0 = None
        loss = 0.0

        # Simular o comportamento ao longo do horizonte
        for t in range(horizon_length):
            delta_t = torch.clamp(delta_controls[t], -0.5, 0.5)  # Limitar δ
            a_t = torch.clamp(a_controls[t], -1.0, 1.0)  # Limitar a

            # Preparar entrada para a RNN
            input_t = torch.tensor(states[-1] + [delta_t.item(), a_t.item()], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Fazer predição com a RNN
            output, (h0, c0) = model(input_t, h0, c0)
            x_dot, y_dot, psi_dot, v_dot = output[0, 0] #clean up

            # Atualizar o estado
            new_x = states[-1][0] + x_dot.item() * dt ## clean up this var x
            new_y = states[-1][1] + y_dot.item() * dt
            new_psi = states[-1][2] + psi_dot.item() * dt
            new_v = states[-1][3] + v_dot.item() * dt
            new_v = max(0.5, min(2.0, new_v))  # Limitar v
            new_psi = np.arctan2(np.sin(new_psi), np.cos(new_psi))

            states.append([new_x, new_y, new_psi, new_v])

            # Calcular custo
            y_error = new_y - y_ref #y_error = new_y
            psi_error = np.arctan2(np.sin(new_psi - psi_ref), np.cos(new_psi - psi_ref))
            #psi_error = np.arctan2(np.sin(new_psi), np.cos(new_psi))
            loss += w_y * (y_error ** 2) + w_psi * (psi_error ** 2) # + Qv*(v[k]-vref)**2

            # Penalizar suavidade dos controles
            if t > 0:
                delta_rate = torch.clamp(delta_rate[t-1], -delta_rate_max*vmax/v, delta_rate_max*vmax/v)
                #vmax=2,52 m/s #amax=3m/s**2
                delta_prev = torch.clamp(delta_controls[t-1], -delta_rate, delta_rate)
                a_prev = torch.clamp(a_controls[t-1], -1.0, 1.0)
                loss += w_delta * ((delta_t - delta_prev) ** 2) + w_acc * (a_t - a_prev) ** 2 #Rdelta, Racceleration

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Retornar os controles otimizados para o primeiro timestep
    delta_opt = torch.clamp(delta_controls[0], -0.5, 0.5).item()
    a_opt = torch.clamp(a_controls[0], -1.0, 1.0).item()

    return delta_opt, a_opt

# Função para processar a imagem da câmera
def process_camera_image(psi_carro, image_width=640, image_height=480, lookahead_distance=0.5, pixel_to_meters_y=0.01):
    """
    Processa uma imagem da câmera para extrair y_ref e ψ_ref.
    Args:
        psi_carro: Yaw atual do carro (rad)
        image_width, image_height: Dimensões da imagem
        lookahead_distance: Distância à frente (m)
        pixel_to_meters_y: Conversão de pixels para metros em y
    Returns:
        y_ref: Offset lateral (m)
        psi_ref: Yaw de referência (rad)
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    a = 0.0003
    b = 0.05
    c = image_width / 2
    for y in range(image_height):
        x_center = int(a * y**2 + b * y + c)
        x_start = max(0, x_center - 10)
        x_end = min(image_width, x_center + 10)
        mask[y, x_start:x_end] = 255

    v_lookahead = int(image_height - (lookahead_distance / pixel_to_meters_y))
    row = mask[v_lookahead]
    lane_pixels = np.where(row == 255)[0]
    if len(lane_pixels) == 0:
        return 0.0, 0.0
    x_ideal = int(np.mean(lane_pixels))
    y_ref = (x_ideal - image_width / 2) * pixel_to_meters_y  # Convertido para metros

    y2 = max(0, v_lookahead - 50)
    row2 = mask[y2]
    lane_pixels2 = np.where(row2 == 255)[0]
    if len(lane_pixels2) == 0:
        psi_ref = 0.0
    else:
        x_ideal2 = int(np.mean(lane_pixels2))
        dx = x_ideal - x_ideal2
        dy = (v_lookahead - y2) * pixel_to_meters_y
        psi_ref = np.arctan2(dx * pixel_to_meters_y, dy)

    return y_ref, psi_ref   #this has to be the state desired or reference: [0,0,vref]
                            #vref is 'current speed' or 'cruise speed', vcurr, vcruise

# Função do loop principal
def main_loop(model, num_steps=100, image_width=640, image_height=480, pixel_to_meters_x=0.001, pixel_to_meters_y=0.01, lookahead_distance=0.5, dt=0.1, L=0.15, device="cpu"):
    """
    Loop principal para controle do carro em tempo real.
    """
    x = 0.0 #not needed state[ey,yaw,velocity] ML_lane_detection(ey, yaw), getSpeed(velocity)
    y = 0.0 #ML_lane_detection
    psi = 0.0 #ML_lane_detection
    v = 1.0 #getSpeed() : measureSpeed() , setSpeed(Vcruise ou Vmanual), loop PID thread
    h0 = None
    c0 = None

    for step in range(num_steps):
        y_ref, psi_ref = process_camera_image(
            psi,
            image_width=image_width,
            image_height=image_height,
            lookahead_distance=lookahead_distance,
            pixel_to_meters_y=pixel_to_meters_y
        )

        current_state = [x, y, psi, v] #has to be changed =>[y,psi,v] = [ey,yaw,velocity]
        #delta_opt : optimal steering
        #a_opt : optimal acceleration : setSpeed(getSpeed() + a_opt*dt)
        delta_opt, a_opt = mpc_step(
            model,
            current_state,
            y_ref, #=0
            psi_ref, #=0
            horizon_length=5,
            dt=dt,
            L=L,
            device=device
        )

        model.eval()
        with torch.no_grad():
            input_t = torch.tensor(current_state + [delta_opt, a_opt], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output, (h0, c0) = model(input_t, h0, c0)
            x_dot, y_dot, psi_dot, v_dot = output[0, 0].cpu().numpy()
            h0 = h0.detach()
            c0 = c0.detach()

        x += x_dot * dt # cleanup
        y += y_dot * dt
        psi += psi_dot * dt
        v += v_dot * dt
        v = np.clip(v, 0.5, 2.0)
        psi = np.arctan2(np.sin(psi), np.cos(psi))

        print(f"Passo {step+1}: x={x:.2f}, y={y:.2f}, ψ={psi:.2f}, v={v:.2f}, δ={delta_opt:.2f}, a={a_opt:.2f}, y_ref={y_ref:.2f}, ψ_ref={psi_ref:.2f}")

    print("Simulação concluída!")

# Função principal
def main():
    input_size = 6  # [x, y, ψ, v, δ, a] 5
    hidden_size = 128
    num_layers = 2
    output_size = 4  # [ẋ, ẏ, ψ̇, v̇] 3
    sequence_length = 5
    num_sequences = 1000
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNForMPC(input_size, hidden_size, num_layers, output_size, sequence_length)
    inputs, targets = generate_training_data(num_sequences, sequence_length)
    print(f"Dados gerados: inputs shape {inputs.shape}, targets shape {targets.shape}")

    train_rnn(model, inputs, targets, num_epochs, batch_size, learning_rate, device)
    torch.save(model.state_dict(), "rnn_for_mpc.pth")
    print("Modelo salvo em 'rnn_for_mpc.pth'")

    model.load_state_dict(torch.load("rnn_for_mpc.pth", map_location=device))
    print("Modelo carregado para inferência.")

    main_loop(model, num_steps=100, device=device)

if __name__ == "__main__":
    main()