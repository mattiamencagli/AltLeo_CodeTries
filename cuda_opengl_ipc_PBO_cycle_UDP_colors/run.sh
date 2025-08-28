#!/bin/bash

# Uscita se qualcosa ritorna un errore
set -e

# Pulizia eventuali server socket esistenti
if [ -e /tmp/cuda_ipc_server ]; then
    echo "Rimuovo vecchio server socket..."
    rm -f /tmp/cuda_ipc_server
fi

# Esci se non trovi tutti gli eseguibili
if [[ ! -f ".sender.moc" || ! -f "receiver.x" || ! -f "sender.x" ]]; then
    echo "Errore: uno o più file richiesti non sono stati trovati." >&2
    exit 1
fi

# Avvia il lettore (OpenGL) in background
echo "Avvio il receiver (OpenGL)..."
if [ -z "$1" ]; then
    ./receiver.x &
elif [ "$1" = "prof" ]; then
    nsys profile --trace=cuda,opengl -f true -o report_opengl_rcv ./receiver.x &
fi

READER_PID=$!

# Aspetta un attimo che il server Qt sia pronto
echo "Aspetto che il server Qt sia pronto..."
sleep 1.0

# Ora avvia il sender (CUDA)
echo "Avvio il sender (CUDA)..."
if [ -z "$1" ]; then
    ./sender.x
elif [ "$1" = "prof" ]; then
    nsys profile --trace=cuda,opengl -f true -o report_cuda_send ./sender.x
fi

# Attendi che il lettore finisca
wait $READER_PID

echo "~~~ The End ~~~"
