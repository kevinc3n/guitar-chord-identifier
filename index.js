var socket = io.connect('http://127.0.0.1:5000');

document.getElementById('startButton').addEventListener('click', function() {
    socket.emit('listen', {data: 'Start listening for audio'});

    console.log('hello')
    document.getElementById('startButton').style.display = 'none';
    document.getElementById('stopButton').style.display = 'inline';
});

document.getElementById('stopButton').addEventListener('click', function() {

    socket.emit('exit', {data: 'Stop listening for audio'});

    document.getElementById('stopButton').style.display = 'none';
    document.getElementById('startButton').style.display = 'inline';
});

socket.on('message', function(message) {
    console.log('Message from server:', message);
});

socket.on('prediction', function(pred) {
    document.getElementById('output').innerHTML = pred;
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
});