from flask import Flask, g
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from AudioRecorder import AudioRecorder

app = Flask(__name__)
CORS(app, origin='*')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route("/")
def ack():
    print("Server acknowledges initial request")
    ar = get_ar()
    return "Server acknowledges initial request"

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('message')
def handle_message(message):
    send(message)

@socketio.on('listen')
def listen(message):
    ar = get_ar()
    while get_rs():
        pred = ar.start_recording()
        socketio.send('Recording started')
        socketio.emit('prediction', pred)
    socketio.send('Stopped listening')

@socketio.on('exit')
def exit():
    set_rs(False)
    socketio.send("Recording stopped")

@socketio.on('disconnect')
def test_disconnect():
    exit()
    print('Client disconnected')

@app.teardown_appcontext
def teardown_ar(exception):
    ar = g.pop('ar', None)

def get_ar():
    if 'ar' not in g:
        g.ar = AudioRecorder()
    return g.ar

def get_rs():
    if 'rs' not in g:
        g.rs = True
    return g.rs

def set_rs(value):
    g.rs = value

if __name__ == "__main__":
    socketio.run(app)
