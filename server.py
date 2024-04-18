from flask import Flask, g, render_template
from flask_socketio import SocketIO, send, emit
import numpy as np
import time
import librosa
import sys
from sklearn.preprocessing import LabelEncoder
from AudioRecorder import AudioRecorder

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route("/")
def ack():
    print("server acknowledges initial request")
    ar = get_ar()
    return "server acknowledges initial request"

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('message')
def handle_message(message):
    send(message)

@socketio.on('listen')
def listen(message):
    ar = get_ar()
    while g.rs == True:
        ar.start_recording()
        socketio.send('recording started')
        pred = ar.consume()
        socketio.send('sending prediction')
        socketio.send(pred)
    socketio.send('stopped listening')

@socketio.on('exit')
def exit():
    g.rs = False
    socketio.send("recording stopped")

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

if __name__ == "__main__":
    socketio.run(app)