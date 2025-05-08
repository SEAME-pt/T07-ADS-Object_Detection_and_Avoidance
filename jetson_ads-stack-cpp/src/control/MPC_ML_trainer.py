import socket
import struct
import numpy as np
import time

# JetRacer parameters
DT = 0.02

def generate_ml_data(current_state):
    """
    Placeholder for ML model.
    Input: current_state = [ey, psi_err, v]
    Output: [ey, psi_err, v, delta]
    """
    ey, psi_err, v = current_state
    # Example values (replace with TensorFlow model)
    ey = ey + 0.01 * np.random.randn()  # Simulated offset
    psi_err = psi_err + 0.01 * np.random.randn()  # Simulated heading error
    delta = 0.02 * np.random.randn()  # Simulated steering angle
    return [ey, psi_err, v, delta]

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 12345))
    server_socket.listen(1)
    print("ML server listening on port 12345...")

    conn, addr = server_socket.accept()
    print(f"Connected to {addr}")

    try:
        while True:
            state_buffer = conn.recv(3 * 8)  # 3 doubles
            if not state_buffer:
                break
            current_state = struct.unpack('3d', state_buffer)

            ml_data = generate_ml_data(current_state)

            packed_data = struct.pack('4d', *ml_data)
            conn.sendall(packed_data)

            time.sleep(DT)
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()