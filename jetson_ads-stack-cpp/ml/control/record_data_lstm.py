import cv2
import numpy as np
import json
import time

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

# Função para obter comandos manuais (simulada, substituir por entrada real)
def get_manual_commands():
    try:
        delta = float(input("Digite o ângulo de viragem (delta, -0.5 a 0.5): "))
        a = float(input("Digite a aceleração (a, -1.0 a 1.0): "))
        return np.clip(delta, -0.5, 0.5), np.clip(a, -1.0, 1.0)
    except ValueError:
        print("Entrada inválida. Usando valores padrão.")
        return 0.0, 0.0

# Configuração
camera = cv2.VideoCapture(0)
servo = Servo()
motor = Motor()
dt = 0.1
pixel_to_meters_y = 0.01
lookahead_distance = 0.5
image_width, image_height = 640, 480
sequence_length = 5

# Buffers para armazenar dados
data = []
sequence_buffer = []

start_time = time.time()
print("Captura de dados iniciada. Forneça comandos manuais por 60 segundos.")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        v = get_velocity()
        y, psi, y_ref, psi_ref, v_ref = get_lane_info(frame, pixel_to_meters_y, lookahead_distance, image_width, image_height, v)
        delta_cmd, a_cmd = get_manual_commands()
        servo.set_angle(delta_cmd)
        motor.set_acceleration(a_cmd)
        
        sequence_buffer.append({
            'y': y,
            'psi': psi,
            'v': v,
            'y_ref': y_ref,
            'psi_ref': psi_ref,
            'v_ref': v_ref,
            'delta': delta_cmd,
            'a': a_cmd
        })
        
        if len(sequence_buffer) > sequence_length:
            sequence_buffer.pop(0)
        
        if len(sequence_buffer) == sequence_length:
            data.append(sequence_buffer.copy())
        
        print(f"y={y:.3f}, psi={psi:.3f}, v={v:.3f}, y_ref={y_ref:.3f}, psi_ref={psi_ref:.3f}, v_ref={v_ref:.3f}, delta={delta_cmd:.3f}, a={a_cmd:.3f}")
        
        time.sleep(max(0, dt - (time.time() - start_time) % dt))
        
        if time.time() - start_time > 60:
            break

finally:
    # Salvar os dados em formato JSON
    with open('real_data_sequences.json', 'w') as f:
        json.dump(data, f)
    print(f"Dados salvos em 'real_data_sequences.json'. Total de sequências: {len(data)}")
    
    camera.release()
    print("Captura de dados terminada!")