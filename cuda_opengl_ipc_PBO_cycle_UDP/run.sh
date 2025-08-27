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
if [ $1 = 'prof' ]; then
    nsys profile --trace=cuda,opengl ./receiver.x &
else
    ./receiver.x &
fi
READER_PID=$!

# Aspetta un attimo che il server Qt sia pronto
echo "Aspetto che il server Qt sia pronto..."
sleep 1.0

# Ora avvia il sender (CUDA)
echo "Avvio il sender (CUDA)..."
if [ $1 = 'prof' ]; then
    nsys profile --trace=cuda,opengl ./sender.x
else
    ./sender.x
fi

# Attendi che il lettore finisca
wait $READER_PID

echo "FINE senza errori."
