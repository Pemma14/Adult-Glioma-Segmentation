# Glioma Segmentation Service

Локальный (on-premise) сервис для автоматической 3D-сегментации диффузных глиом взрослых с интеграцией в TPS (Treatment Planning System).

Предназначен для врачей-радиационных онкологов, работающих в системах планирования лучевой терапии. Он автоматически выделяет три региона опухоли (Whole Tumour / Tumour Core / Enhancing Tumour — WT/TC/ET) на МРТ-снимках головного мозга, рассчитывает их объёмы и формирует структуры для импорта в TPS в формате RTSTRUCT. Поддерживается интеграция с Varian Eclipse через ESAPI-плагин, а также загрузка исследований в форматах NIfTI и DICOM.

## Архитектура

```
Eclipse (плагин ESAPI) ── REST ──→ FastAPI ── RabbitMQ ──→ Worker 
                                     │                    │
                                  PostgreSQL         Model registry
```

## Структура проекта

| Директория | Назначение |
|---|---|
| `app/` | FastAPI-сервер, маршруты, сервисы, ORM-модели, DICOM-шлюз |
| `ml_worker/` | Inference-worker на RabbitMQ |
| `src/glioma/` | Продакшен-пайплайн инференса |
| `src/models/` | Архитектура модели SwinUNETR |
| `models_registry/` | Версионированные чекпоинты моделей + конфигурации |
| `plugin-eclipse/` | C#-плагин ESAPI для Varian Eclipse |
| `nginx/` | Конфигурация reverse proxy и SSL |
| `tests/` | Тесты (smoke-тесты API и др.) |
| `experiments/` | Исследования и обучение (отдельное рабочее пространство) |
| `docs/` | Бизнес-анализ и прототип продукта |
| `data/` | Рабочие данные сервиса: загрузки, результаты, временные DICOM-файлы (создаётся при работе, не версионируется) |

### Основные файлы в корне

| Файл | Назначение |
|---|---|
| `docker-compose.yml` | Описание всего стека сервисов для Docker Compose |
| `pyproject.toml` | Зависимости Python, метаданные и настройки инструментов |
| `pytest.ini` | Конфигурация тестового фреймворка pytest |
| `.env.example` | Пример переменных окружения |
| `AI_RULES.md` | Правила и соглашения для AI-ассистентов |

## Окружение и зависимости

Проект использует [`uv`](https://docs.astral.sh/uv/) для управления виртуальным окружением и зависимостями.

```bash
# Создать виртуальное окружение и установить зависимости
uv sync

# Активировать окружение (опционально)
source .venv/bin/activate

# Запускать команды можно и без ручной активации через uv run
uv run python -m app.main
```

При запуске через `uv run` используется виртуальное окружение из директории `.venv/`, управляемое `uv`.

## Быстрый старт

```bash
# 1. Настройка
cp .env.example .env
# отредактируй .env: установите AUTH__API_KEY, GLIOMA__MODEL_VERSION

# 2. Запуск всего стека
docker compose up --build
```

Или без Docker:

```bash
uv sync
# терминал 1
uv run python -m app.main
# терминал 2
uv run python -m ml_worker.glioma_worker
```

## API

Все запросы требуют заголовок `X-API-Key`.

```bash
# Загрузить NIfTI
curl -X POST localhost:8500/api/v1/segmentation/upload \
  -H "X-API-Key: your-key" \
  -F "file=@scan.nii.gz"

# Загрузить DICOM (zip)
curl -X POST localhost:8500/api/v1/segmentation/from-dicom \
  -H "X-API-Key: your-key" \
  -F "file=@dicom.zip"

# Проверить статус
curl localhost:8500/api/v1/segmentation/status/1 \
  -H "X-API-Key: your-key"

# Получить результат с объёмами и ссылками для скачивания
curl localhost:8500/api/v1/segmentation/result/1 \
  -H "X-API-Key: your-key"

# Скачать RTSTRUCT (для импорта в TPS)
curl -o rtstruct.dcm localhost:8500/api/v1/segmentation/1/rtstruct \
  -H "X-API-Key: your-key"
```

## Веб-интерфейс

- `/viewer/app.html` — загрузка файла, отслеживание статуса, просмотр объёмов, скачивание результатов (в проде будет автоматически экспортироваться в TPS)
- `/viewer/viewer.html` — интерактивный просмотрщик NIfTI/DICOM с оверлеем

## CLI-инференс (автономно, без сервера)

```bash
uv run python -m src.glioma --image scan.nii.gz --output_dir ./results
```

## Интеграция с Eclipse

Директория `plugin-eclipse/` содержит C#-скрипт ESAPI, который:
- экспортирует DICOM-серию из текущего плана,
- отправляет её в API сегментации,
- скачивает RTSTRUCT и импортирует его в набор структур.

## Тесты

```bash
uv run pytest tests/ -v
```

## Реестр моделей

```
models_registry/v1.0.0/
├── config.yaml
├── best_model_swin_unetr_fold{0..4}.pth
```
