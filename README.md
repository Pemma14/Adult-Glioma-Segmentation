# Glioma Segmentation Service

On-premise service for automated 3D segmentation of adult-type diffuse gliomas (WT/TC/ET) with TPS integration.

## Architecture

```
Eclipse (ESAPI plugin) ──REST──→ FastAPI ──RabbitMQ──→ Worker (SwinUNETR)
                                     │                      │
                                  PostgreSQL            models_registry/
```

## Structure

| Directory | Purpose |
|---|---|
| `app/` | FastAPI server, routes, services, ORM models, DICOM gateway |
| `ml_worker/` | RabbitMQ inference worker |
| `src/glioma/` | Production inference pipeline |
| `src/models/` | Neural architectures (SwinUNETR, BaselineUNet, SwinDER3D) |
| `models_registry/` | Versioned model checkpoints + config |
| `plugin-eclipse/` | ESAPI C# plugin for Varian Eclipse |
| `experiments/` | Research & training (separate workspace) |
| `docs/` | Business analysis, product prototype |

## Quick start

```bash
# 1. Configure
cp .env.example .env
# edit .env: set AUTH__API_KEY, GLIOMA__MODEL_VERSION

# 2. Run full stack
docker compose up --build
```

Or without Docker:

```bash
uv sync
# terminal 1
uv run python -m app.main
# terminal 2
uv run python -m ml_worker.glioma_worker
```

## API

All requests require `X-API-Key` header.

```bash
# Upload NIfTI
curl -X POST localhost:8500/api/v1/segmentation/upload \
  -H "X-API-Key: your-key" \
  -F "file=@scan.nii.gz"

# Upload DICOM (zip)
curl -X POST localhost:8500/api/v1/segmentation/from-dicom \
  -H "X-API-Key: your-key" \
  -F "file=@dicom.zip"

# Poll status
curl localhost:8500/api/v1/segmentation/status/1 \
  -H "X-API-Key: your-key"

# Get result with volumes + download URLs
curl localhost:8500/api/v1/segmentation/result/1 \
  -H "X-API-Key: your-key"

# Download RTSTRUCT (for TPS import)
curl -o rtstruct.dcm localhost:8500/api/v1/segmentation/1/rtstruct \
  -H "X-API-Key: your-key"
```

## Web UI

- `/viewer/app.html` — upload file, track status, view volumes, download results
- `/viewer/viewer.html` — interactive NIfTI/DICOM viewer with overlay

## CLI inference (standalone, no server)

```bash
uv run python -m src.glioma --image scan.nii.gz --output_dir ./results
```

## Eclipse integration

The `plugin-eclipse/` folder contains an ESAPI C# script that:
- Exports DICOM series from the current plan
- Sends to the segmentation API
- Downloads RTSTRUCT and imports it into the structure set

## Tests

```bash
uv run pytest tests/ -v
```

## Model registry

```
models_registry/v1.0.0/
├── config.yaml
├── best_model_swin_unetr_fold{0..4}.pth
```
