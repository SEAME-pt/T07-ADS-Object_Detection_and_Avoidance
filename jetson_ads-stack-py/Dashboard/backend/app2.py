import zmq
import threading
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import os

app = Flask(__name__, template_folder='../frontend/')
socketio = SocketIO(app)

# Create context
context_main = zmq.Context()

# Sensor socket (5555) Main
sensor_socket = context_main.socket(zmq.SUB)
sensor_socket.connect("tcp://localhost:5555")
sensor_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Listen for sensors
def listen_for_sensor_data():
    # Configure the poller
    poller = zmq.Poller()
    poller.register(sensor_socket, zmq.POLLIN)  # Register sensor_socket for reading

    while True:
        # Initialize the message variable
        message_sensor = None

        # Wait for events on registered sockets
        events = dict(poller.poll())

        # Process messages from sensor_socket
        if sensor_socket in events:
            message_sensor = sensor_socket.recv_string()
            print(f"Sensor Data: {message_sensor}")

        # Create a dictionary to store received data
        full_data = {}

        # Process data from sensor_socket, if available
        if message_sensor:
            parts = message_sensor.split(' ')
            if len(parts) == 2:
                sensor_type, sensor_value = parts
                full_data[sensor_type] = sensor_value

        # Emit data if there is something to send
        if full_data:
            socketio.emit('vehicle_sensors', full_data)

# Start thread to listen for ZeroMQ data
sensor_thread = threading.Thread(target=listen_for_sensor_data, daemon=True)
sensor_thread.start()

@app.route('/')
def index():
    return render_template('index.html')  # index HTML template

# API for vehicle info
@app.route('/vehicle-info', methods=['GET'])
def get_vehicle_info():
    json_file_path = '../../MainCar/vehicle_info.json'  # Path to the generated JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return jsonify(data)
    else:
        return jsonify({"error": "Vehicle information not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # Flask running on port 5000
