import cv2
import numpy as np
import torch
import time

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

# Carregar o modelo treinado
sequence_length = 5
model = LSTMModel(input_size=6, hidden_size=64, output_size=2, num_layers=1)
try:
    model.load_state_dict(torch.load("lstm_model_real_data.pth"))
    print("Modelo treinado carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O ficheiro 'lstm_model_real_data.pth' não foi encontrado. Treina o modelo primeiro.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Função para prever comandos
def lstm_model_step(model, sequence, h0, c0, device="cpu"):
    model.eval()
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        controls, (h0, c0) = model(sequence_tensor, (h0, c0))
    delta_opt = torch.clamp(controls[0, 0], -0.5, 0.5).item()
    a_opt = torch.clamp(controls[0, 1], -1.0, 1.0).item()
    return delta_opt, a_opt, h0, c0

# Simulação de atuadores
class Servo:
    def __init__(self):
        self.angle = 0.0
    def set_angle(self, angle):
        self.angle = np.clip(angle, -0.5, 0.5)
    def get_angle(self):
        return self.angle

class Motor:
    def __init__(self):
        self.accel = 0.0
    def set_acceleration(self, accel):
        self.accel = np.clip(accel, -1.0, 1.0)
    def get_acceleration(self):
        return self.accel

# Função para segmentação com U-Net (simulada, substituir pela tua U-Net)
def unet_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask

# Função para obter informações da faixa
def get_lane_info(image, pixel_to_meters_y=0.01, lookahead_distance=0.5, image_width=640, image_height=480, v=1.0):
    mask = unet_segmentation(image)
    points = np.where(mask == 255)
    y_pixels = points[0]
    x_pixels = points[1]
    
    if len(y_pixels) < 10:
        return 0.0, 0.0, 0.0, 1.0
    
    current_y_line = image_height - 10
    current_x_pixels = x_pixels[y_pixels >= current_y_line]
    y = (np.mean(current_x_pixels) - image_width / 2) * pixel_to_meters_y if current_x_pixels.size > 0 else 0.0
    
    coeffs = np.polyfit(y_pixels, x_pixels, 2)
    poly = np.poly1d(coeffs)
    
    poly_deriv = poly.deriv()
    dx_dy = poly_deriv(current_y_line)
    psi = np.arctan(dx_dy)
    
    y_lookahead = image_height - (lookahead_distance / pixel_to_meters_y)
    x_ref = poly(y_lookahead)
    y_ref = (x_ref - image_width / 2) * pixel_to_meters_y
    dx_dy_ref = poly_deriv(y_lookahead)
    psi_ref = np.arctan(dx_dy_ref)
    poly_deriv2 = poly_deriv.deriv()
    curvature = abs(poly_deriv2(y_lookahead))
    v_ref = 1.0 if curvature < 0.01 else 0.5
    
    return y, psi, y_ref, psi_ref, v_ref

# Função para obter velocidade (substituir pela tua fonte real)
def get_velocity():
    return 1.0

# Configuração
camera = cv2.VideoCapture(0)
servo = Servo()
motor = Motor()
dt = 0.1
pixel_to_meters_y = 0.01
lookahead_distance = 0.5
image_width, image_height = 640, 480

# Buffers para sequência e filtragem
sequence_buffer = []
y_buffer = []
psi_buffer = []
v = 1.0

# Inicializar estados ocultos
h0 = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)
c0 = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)

start_time = time.time()
print("Condução autónoma iniciada. O carro será controlado por 60 segundos.")

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    v = get_velocity()
    y, psi, y_ref, psi_ref, v_ref = get_lane_info(frame, pixel_to_meters_y, lookahead_distance, image_width, image_height, v)
    
    y_buffer.append(y)
    psi_buffer.append(psi)
    if len(y_buffer) > 5:
        y_buffer.pop(0)
        psi_buffer.pop(0)
    y_filtered = np.mean(y_buffer)
    psi_filtered = np.mean(psi_buffer)
    
    sequence_buffer.append([y_filtered, psi_filtered, v, y_ref, psi_ref, v_ref])
    if len(sequence_buffer) > sequence_length:
        sequence_buffer.pop(0)
    
    if len(sequence_buffer) == sequence_length:
        delta_cmd, a_cmd, h0, c0 = lstm_model_step(model, sequence_buffer, h0, c0, device)
    else:
        delta_cmd, a_cmd = 0.0, 0.0
    
    if abs(y_filtered) > 0.5:
        delta_cmd = -np.sign(y_filtered) * 0.5
    
    servo.set_angle(delta_cmd)
    motor.set_acceleration(a_cmd)
    
    print(f"y={y_filtered:.3f}, psi={psi_filtered:.3f}, v={v:.3f}, y_ref={y_ref:.3f}, psi_ref={psi_ref:.3f}, v_ref={v_ref:.3f}, delta={delta_cmd:.3f}, a={a_cmd:.3f}")
    
    time.sleep(max(0, dt - (time.time() - start_time) % dt))
    
    if time.time() - start_time > 60:
        break

camera.release()
print("Condução autónoma terminada!")