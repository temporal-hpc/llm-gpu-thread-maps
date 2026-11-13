#!/bin/bash

# =================================================================
# Energy of the LLM inference: runs one inference per (model, domain, samples)
# cell, one model at a time, while measure_multigpu.py logs the power of all
# visible GPUs. Needs ollama, curl, jq and bc. Run from the repository root:
#     bash inference_energy/run_inference_energy.sh
# CONFIGURATION
# =================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERMON_CMD="python3 $HERE/measure_multigpu.py"
TEMP_POWER_LOG="power-gpu_log.dat"

# Models
MODELS=(
    "nemotron:70b"
    "mistral-nemo"
    "llama4"
    "llama3.3"
    "gpt-oss:120b"
    "gpt-oss:20b"
    "gemma3:27b"
    "gemma3:12b"
    "deepseek-r1:70b"
    "qwen3:32b"
    "qwen3:235b"
)

PROMPT_TEMPLATE_PATH="prompt_template.txt"
DOMAINS_DIR="domains"
BASE_RESULTS_PATH="results"
STAGES=(20 50 100)

# =================================================================
# START
# =================================================================
echo "=== inference energy: full grid ==="
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_PATH="$BASE_RESULTS_PATH/$TIMESTAMP"
ENERGY_LOGS_PATH="$RESULTS_PATH/energy_logs"

mkdir -p "$RESULTS_PATH"
mkdir -p "$ENERGY_LOGS_PATH"
CSV_RESULTS_FILE="$RESULTS_PATH/benchmark_summary.csv"

echo "Date,Domain,Model,Stage_Lines,ElapsedTime_s,Status,ErrorMessage,EnergyFile" > "$CSV_RESULTS_FILE"

# Basic checks
if [ ! -d "$DOMAINS_DIR" ]; then echo "Error: missing $DOMAINS_DIR (run download_domains.sh)"; exit 1; fi

# Start Ollama
ollama serve &
OLLAMA_PID=$!
while ! curl --silent --fail http://127.0.0.1:11434 > /dev/null; do sleep 1; done
echo ">> Ollama ready."

# =================================================================
# MAIN LOOP (all domains)
# =================================================================
for domain_file in "$DOMAINS_DIR"/*.txt; do
    DOMAIN_NAME=$(basename "$domain_file" .txt)
    echo ">> DOMAIN: $DOMAIN_NAME"
    
    DOMAIN_RESULTS_PATH="$RESULTS_PATH/$DOMAIN_NAME"
    mkdir -p "$DOMAIN_RESULTS_PATH"
    
    for model in "${MODELS[@]}"; do
        for line_count in "${STAGES[@]}"; do
            echo "   Model: $model | Samples: $line_count"
            
            # Prompt
            PROMPT_TEMPLATE=$(cat "$PROMPT_TEMPLATE_PATH")
            DOMAIN_DATA=$(head -n "$line_count" "$domain_file")
            PROMPT_CONTENT="${PROMPT_TEMPLATE/__MAPPING_DATA_HERE__/$DOMAIN_DATA}"
            
            SAFE_MODEL_NAME=$(echo "$model" | tr ':' '_')
            OUTPUT_FILE="$DOMAIN_RESULTS_PATH/response_${SAFE_MODEL_NAME}_${line_count}_lines.txt"
            
            # JSON Payload
            JSON_PAYLOAD=$(jq -n --arg model "$model" --arg prompt "$PROMPT_CONTENT" '{model: $model, prompt: $prompt, stream: false}')
            echo "$JSON_PAYLOAD" > temp_payload.json

            # Request script, so that the monitor wraps exactly one inference
            echo "#!/bin/bash" > _request.sh
            echo "curl -s --max-time 3600 -X POST http://127.0.0.1:11434/api/generate -d @temp_payload.json > temp_response.json" >> _request.sh
            chmod +x _request.sh

            # MEASUREMENT
            start_time=$(date +%s.%N)
            $POWERMON_CMD ./_request.sh
            end_time=$(date +%s.%N)
            ELAPSED_TIME=$(echo "$end_time - $start_time" | bc)

            # Response
            if [ -f "temp_response.json" ]; then RESPONSE_JSON=$(cat temp_response.json); else RESPONSE_JSON='{"error": "No response"}'; fi
            
            # Energy log
            ENERGY_FILENAME="energy_${DOMAIN_NAME}_${SAFE_MODEL_NAME}_st${line_count}.csv"
            if [ -f "$TEMP_POWER_LOG" ]; then mv "$TEMP_POWER_LOG" "$ENERGY_LOGS_PATH/$ENERGY_FILENAME"; else ENERGY_FILENAME="N/A"; fi

            # Retry once if the model had to be pulled
            if echo "$RESPONSE_JSON" | jq -e '.error' | grep -q "not found"; then
                echo "   Pulling model..."
                curl -s -X POST http://127.0.0.1:11434/api/pull -d '{ "name": "'"$model"'" }'
                $POWERMON_CMD ./_request.sh
                RESPONSE_JSON=$(cat temp_response.json)
                if [ -f "$TEMP_POWER_LOG" ]; then mv "$TEMP_POWER_LOG" "$ENERGY_LOGS_PATH/$ENERGY_FILENAME"; fi
            fi

            TODAY_DATE=$(date +%Y-%m-%d)
            if echo "$RESPONSE_JSON" | jq -e '.error' > /dev/null; then
                ERROR_MSG=$(echo "$RESPONSE_JSON" | jq -r '.error' | sed 's/"/""/g')
                echo "$TODAY_DATE,$DOMAIN_NAME,$model,$line_count,0,FAILURE,\"$ERROR_MSG\",$ENERGY_FILENAME" >> "$CSV_RESULTS_FILE"
            else
                echo "$RESPONSE_JSON" | jq -r '.response' > "$OUTPUT_FILE"
                echo "$TODAY_DATE,$DOMAIN_NAME,$model,$line_count,$ELAPSED_TIME,SUCCESS,,$ENERGY_FILENAME" >> "$CSV_RESULTS_FILE"
            fi
            
            rm -f temp_payload.json temp_response.json _request.sh
            sleep 3
        done
    done
done

kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo ">> Done."