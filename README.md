# Finding Parallel GPU Thread Maps with Open Weight LLMs

Code and data to reproduce the experiments of the manuscript
*Finding Parallel GPU Thread Maps with Open Weight LLMs*, by Jose Maureira,
Cristóbal A. Navarro, Hector Ferrada, Luis Veas-Castillo and Guillermo Campos.

The pipeline samples the first points of a non-box-shaped domain, asks an
open-weight LLM for a function `map_to_coordinates(n)` that maps a linear
index to a coordinate of the domain, validates the function against a ground
truth of 10^6 points, and measures the resulting thread map as a CUDA kernel
against the bounding-box approach.

## Contents

```
prompt_template.txt     prompt shared by all models and domains
download_domains.sh     fetches the six ground-truth domains into domains/
inference/              the 1,188 inferences (Ollama)
responses/              the 1,188 model outputs, verbatim, failures included
evaluation/             syntactic repair, Ordered and Domain Coverage graders,
                        pass@k, tables
baselines/              symbolic-regression baseline (gplearn)
inference_energy/       energy of the LLM inference and its figure
gpu/                    CUDA block-level benchmark and power trace
data/                   results reported in the manuscript
```

### Domains

| File | Domain | Set |
|---|---|---|
| `2D_Triangular.txt` | 2D triangular | validation |
| `2D_Triangulo_Sierpinski.txt` | 2D Sierpinski gasket | validation |
| `2D_Alfombra_Sierpinski.txt` | 2D Sierpinski carpet | validation |
| `3D_Triangular.txt` | 3D pyramid (tetrahedral) | validation |
| `3D_Piramide_Sierpinski.txt` | 3D Sierpinski pyramid | exploration |
| `3D_Menger_Sponge.txt` | 3D Menger sponge | exploration |

Each file lists 10^6 points in order, one per line, as
`Thread  <n> -> Coordenadas: (x, y[, z])`. The first 20, 50 or 100 lines are
what the model sees; the full file is the ground truth. They are distributed
as assets of the [v1.0 release](../../releases/tag/v1.0):

```bash
bash download_domains.sh
```

### Responses

`responses/<domain>__<model>__st<samples>__<rep>.txt` is the text returned by
the model, untouched. There are 11 models x 6 domains x 3 sample counts = 198
cells, and each cell has six files: `1` to `5` are samples under the default
decoding parameters of the model and `greedy` is one pass at temperature 0.
Every output is kept, including those that are empty, contain only prose, do
not parse or return a wrong mapping. The prompt of a response is
`prompt_template.txt` with the first `<samples>` lines of the domain file in
place of `__MAPPING_DATA_HERE__`.

### Data

| File | Content |
|---|---|
| `data/passk_responses.csv` | verdict of each response before and after the syntactic repair |
| `data/passk_cells.csv` | correct samples out of 5, greedy pass and pass@k per cell |
| `data/coverage.csv` | Domain Coverage of each response |
| `data/inference_energy.csv` | energy, mean power and time of the inference per cell |
| `data/sr_baseline_results.json` | output of the symbolic-regression baseline |
| `data/gpu_benchmark_A100_raw.txt` | output of the block-level benchmark on the A100 |
| `data/power_trace.csv` | power trace of the bounding box and the map, 2D triangular |

`data/gpu_benchmark_A100_raw.txt` is the unmodified output of the run
reported in the manuscript. Its `Tot Energy` column is the mean power times
the time of 20 launches, in mJ; the energy of one launch, which is what the
manuscript and the current `gpu/gpu_benchmark.cu` report, is that value
divided by 20,000 (in J). Its row labels are the older ones (for instance
`GPT-OSS:120b (100pts)` is `OSS:120b (100 Samples)`), and it has two rows of
the 2D Sierpinski gasket that are not reported.

## Requirements

- Inference: NVIDIA GPUs and [Ollama](https://ollama.com) **0.12.6**. The
  manuscript used 4x A100 40 GB; `qwen3:235b` alone needs 142 GB.
- Evaluation: Python >= 3.9, standard library only.
- Figures and baseline: `pip install pandas matplotlib seaborn gplearn numpy`.
- GPU benchmark: CUDA Toolkit with `nvcc` and NVML. About 16.5 GB of GPU memory.

The `.sbatch` files are the SLURM jobs used on the Patagón supercomputer,
where every step runs inside a container. They are a reference: partition
names and resources will differ on another cluster.

## Reproduction

### 1. Inference

```bash
ollama pull deepseek-r1:70b && ollama pull gemma3:12b && ollama pull gemma3:27b
ollama pull gpt-oss:20b && ollama pull gpt-oss:120b && ollama pull llama3.3
ollama pull llama4 && ollama pull mistral-nemo && ollama pull nemotron:70b
ollama pull qwen3:32b && ollama pull qwen3:235b

python3 inference/make_payloads.py --all   # one request per inference in payloads/
sbatch inference/inference.sbatch          # or: ollama serve & bash inference/drive.sh
python3 inference/parse_raw.py             # raw/ -> responses/
```

Models run with the decoding parameters of their Ollama `Modelfile`; no
option is passed except `temperature = 0` in the greedy pass. Sampling is
stochastic, so a new run gives different responses from those in
`responses/`. Steps 2 and 3 work on the archived responses and do not need a
GPU.

### 2. Evaluation

```bash
bash download_domains.sh
bash evaluation/grade_all.sh               # or: sbatch evaluation/evaluation.sbatch
```

This grades every response three times and rewrites `data/passk_*.csv` and
`data/coverage.csv`:

- **Ordered**: `map_to_coordinates(n)` must equal the ground truth for every
  `n` in [0, 10^6). The grader aborts at the first mismatch.
- **Domain Coverage**: fraction of the distinct coordinates of the domain
  produced over [0, 10^5), regardless of order.
- **Raw and repaired**: Ordered is computed on the code as emitted and after
  the syntactic repair of `evaluation/repair.py`. The repair isolates the code
  from the surrounding prose, aliases the mapping function when it has another
  name and drops module-level test statements. It never alters an expression,
  a constant, an operator or the control flow.

The Ordered grader stops a response after 20 minutes and the Domain Coverage
grader after 10 minutes, and a response that is stopped counts as a failure.
Of the 1,188 responses, 13 reach the limit of the Ordered grader. Eleven of
them never return from one of the first two indices. The other two, both of
the 2D Sierpinski gasket with 50 samples (`gpt-oss:120b` and `gpt-oss:20b`,
sample 1), are correct but slow: in the run reported in the manuscript they
were stopped after more than 940,000 correct indices, and on a faster
processor they finish and pass. A new run can therefore give 5/5 and 3/5 in
those two cells, where the manuscript reports 4/5 and 2/5, and 160 exact
responses instead of 158. Every other verdict is the same. The files
`data/passk_*.csv` are the run reported in the manuscript; `TIME_LIMIT`
(seconds, 0 for no limit) changes the limit of the Ordered grader. In the
Domain Coverage grader 20 responses do not finish within the limit and are
recorded as `EVAL_TIMEOUT` with coverage 0 in `data/coverage.csv`. None of
them is the best sample of its cell, so the tables are not affected.

Generated code is executed with `exec`, and a few responses allocate several
GB of memory. Run the graders in a container or a disposable environment.

```bash
python3 evaluation/repair_summary.py       # verdicts before and after the repair
python3 evaluation/make_tables.py          # accuracy tables per domain and pass@k summary
python3 evaluation/menger_taxonomy.py      # structure of the code generated for the Menger sponge
```

### 3. Symbolic-regression baseline

```bash
python3 baselines/sr_baseline.py
```

### 4. Energy of the inference

`inference_energy/run_inference_energy.sh` runs one inference per cell, one
model at a time, wrapped by `measure_multigpu.py`, which polls `nvidia-smi`
every 100 ms and integrates the power of all visible GPUs. The per-inference
logs of the reported run are in the release asset
`inference_energy_logs.tar.gz`.

```bash
bash inference_energy/run_inference_energy.sh
python3 inference_energy/aggregate_energy.py results/<timestamp>/energy_logs
python3 inference_energy/plot_efficiency.py    # figure from data/inference_energy.csv
```

### 5. GPU benchmark

```bash
cd gpu
nvcc -O3 -std=c++14 -o gpu_benchmark gpu_benchmark.cu -lnvidia-ml -lpthread
nvcc -O3 -std=c++14 -o power_trace power_trace.cu -lnvidia-ml -lpthread
./gpu_benchmark        # writes gpu_benchmark_results.txt
./power_trace          # writes power_trace.csv
python3 plot_power_trace.py power_trace.csv
```

`gpu_benchmark.cu` maps N = 5x10^8 points at block level (256 threads per
block, 1,953,125 blocks) and compares each map with the bounding box, which
launches the full enclosing grid and discards the blocks outside the domain.
The kernels of the model rows are hand translations to CUDA of the Python
functions the models emitted. Each row launches its kernel repeatedly for at
least 3 s, and never fewer than 10 times, while NVML integrates the energy of
the GPU. Time is the mean per launch, power is the mean over the measurement,
and energy is the energy of one launch.

`power_trace.cu` runs the bounding box and the map of the 2D triangular
domain 2,000 launches each, with idle intervals in between, sampling the
power every 20 ms.

Timings and power depend on the GPU. The values of the manuscript were
measured on an A100 40 GB SXM4.

## Contact

Jose Maureira, Instituto de Informática, Universidad Austral de Chile.
