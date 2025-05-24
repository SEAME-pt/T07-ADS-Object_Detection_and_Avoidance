import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Iniciando assinante ZMQ...")
try:
    while True:
        message = socket.recv_string()
        print(f"Recebido: {message}")
except KeyboardInterrupt:
    print("Encerrando assinante ZMQ...")
    socket.close()
    context.term()