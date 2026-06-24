# Google Colab GPU Demo Setup

This guide explains how to run the glioma segmentation service on a Google Colab GPU runtime using `notebooks/Colab_Glioma_Demo.ipynb`.

## What is this for?

- Fast GPU-accelerated inference for demos and debugging.
- Testing the Web UI and the interactive NiiVue viewer without a local GPU.
- Quick validation of model weights.

## What is this NOT for?

- Clinical production use.
- Long-running or multi-user deployments.
- Processing real patient data (Colab is a third-party cloud environment).

## Prerequisites

1. A Google account with access to Google Colab.
2. Model weights in `models_registry/v1.0.0/` (upload them to Google Drive or to the Colab Files panel).
3. (Optional) A free [ngrok](https://ngrok.com/) account for a public URL.

## Quick start

1. Open `notebooks/Colab_Glioma_Demo.ipynb` in Colab.
2. Select a GPU runtime: `Runtime → Change runtime type → Hardware accelerator → GPU`.
3. Run the cells in order:
   - Check GPU availability.
   - Clone/upload the project.
   - Install CUDA PyTorch and project dependencies.
   - Copy model weights from Google Drive (or upload them).
   - Start `app.main_colab`.
   - Expose the service via ngrok.
   - Open the printed viewer URL in a browser.

## How it works

- `app/main_colab.py` is a single-process FastAPI app.
- It loads the ensemble **once at startup**, so each inference request reuses the already-loaded models.
- It does **not** use Postgres, RabbitMQ, or background workers.
- It implements the same endpoints as the full service, so `app/static/app.html` works unchanged.

## Limitations

- Colab sessions are ephemeral; you must re-upload/re-download models each session.
- Free Colab GPUs have time limits and may disconnect after inactivity.
- ngrok free URLs are temporary and change on every run.
- Inference blocks the upload request; only one segmentation runs at a time.

## Troubleshooting

### `CUDA out of memory`

Reduce `sw_batch_size` in `models_registry/v1.0.0/config.yaml`, or use a Colab runtime with more VRAM (Pro/Pro+).

### `models_registry/v1.0.0/config.yaml` not found

Make sure you copied the `models_registry` folder into the project root in Colab and set `GLIOMA__MODEL_VERSION=v1.0.0`.

### ngrok tunnel fails

Check that you replaced `YOUR_NGROK_TOKEN` with your actual authtoken.
