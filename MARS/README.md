## Downloading HF resources
> Remember to store on pod's persistent volume mounted at `/workspace`.

- Download `huggingface-cli`: `pip install -U "huggingface_hub[cli]"`.
- Login with HF token before downloading for gated resources (`huggingface-cli login`).
- Download model: `huggingface-cli download --local-dir /workspace/models/<local_model_id> <model_id_hf>`.

Resources directory structure:
- Models: `/workspace/models`
- Datasets: `/workspace/datasets`

## Experiments

1. PCA / tSNE on activations at ${l_i, ..., l_n}$, where $l_i \in$ residual stream transformer layers, on settings:
- Prompt only
- Full generation (prompt + generated answer)
- (opt) CoT setting: (prompt + CoT, but no answer)
    - Potential limitation: how to detect end of CoT / start of answer in model generations

Evaluation: compare model generations against ground truth to obtain labels. Visualize plots in search of clusters across $n$ layers with clear separation among correct / incorrect generations.