./bin/ollama serve &

pid=$!

sleep 5

echo "pulling gemma3n model..."
ollama pull gemma3n:latest

wait $pid