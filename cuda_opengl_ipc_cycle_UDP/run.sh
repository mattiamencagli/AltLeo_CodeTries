#!/bin/bash

# Render with Nvidia GPU
# export __NV_PRIME_RENDER_OFFLOAD=1
# export __GLX_VENDOR_LIBRARY_NAME=nvidia
# export __VK_LAYER_NV_optimus=1

# Uscita se qualcosa ritorna un errore
set -e

# Pulizia eventuali server socket esistenti
if [ -e /tmp/cuda_ipc_server ]; then
    echo "Rimuovo vecchio server socket..."
    rm -f /tmp/cuda_ipc_server
fi

# Avvia il lettore (OpenGL) in background
echo "Avvio il receiver (OpenGL)..."
./receiver.x &
READER_PID=$!

# Aspetta un attimo che il server Qt sia pronto
echo "Aspetto che il server Qt sia pronto..."
sleep 1.0

# Ora avvia il sender (CUDA)
echo "Avvio il sender (CUDA)..."
./sender.x

# Attendi che il lettore finisca
wait $READER_PID

echo "FINE."
