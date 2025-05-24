import zmq
import time
import random

# Configura o contexto e o socket ZMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Mesma porta usada no ZMQReader

# Função para enviar mensagem com um tópico e valor
def send_message(topic, value):
    message = f"{topic} {value}"  # Usa espaço em vez de dois-pontos
    socket.send_string(message)
    print(f"Enviado: {message}")

# Simulação dos sinais
def simulate_signals():
    speed = 0
    total_distance = 0
    battery_percentage = 100
    increasing = True

    while True:
        # Simula velocidade (0 a 100, incrementando/decrementando)
        if increasing:
            speed += 5
            if speed >= 100:
                increasing = False
        else:
            speed -= 5
            if speed <= 0:
                increasing = True
        send_message("speed", str(speed))

        # Simula distância total (aumenta gradualmente)
        total_distance += 0.1
        # send_message("totaldistance", f"{total_distance:.1f}")

        # Simula porcentagem da bateria (diminui lentamente)
        battery_percentage = max(0, battery_percentage - 0.5)
        send_message("batterypercentage", f"{battery_percentage:.1f}")

        # Simula bateria (valor bruto, por exemplo, volts)
        # send_message("battery", f"{battery_percentage * 0.12:.2f}")

        # Simula sinais booleanos (alternando true/false)
        send_message("lineleft", "true" if random.random() > 0.5 else "false")
        send_message("lineright", "true" if random.random() > 0.5 else "false")
        send_message("lightshigh", "true" if random.random() > 0.5 else "false")
        send_message("brake", "true" if random.random() > 0.5 else "false")
        send_message("lightsleft", "true" if random.random() > 0.5 else "false")
        send_message("lightsright", "true" if random.random() > 0.5 else "false")
        # send_message("lightsemergency", "true" if random.random() > 0.5 else "false")
        send_message("lka", "true" if random.random() > 0.5 else "false")
        send_message("autopilot", "true" if random.random() > 0.5 else "false")
        # send_message("horn", "true" if random.random() > 0.9 else "false")
        send_message("lightslow", "true" if random.random() > 0.5 else "false")
        # send_message("lightspark", "true" if random.random() > 0.5 else "false")
        # send_message("ismoving", "true" if speed > 0 else "false")

        # Aguarda 1 segundo antes de enviar o próximo conjunto de mensagens
        time.sleep(.5)

if __name__ == "__main__":
    print("Iniciando publicador ZMQ...")
    time.sleep(1)  # Aguarda conexão do assinante
    try:
        simulate_signals()
    except KeyboardInterrupt:
        print("Encerrando publicador ZMQ...")
        socket.close()
        context.term()