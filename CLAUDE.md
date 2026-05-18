# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Microservices platform for detecting man-made objects (ships) in satellite imagery. Users define geographic polygons, fetch Sentinel-2 imagery for those areas, and run ML-based object detection on the saved images.

## Running Services

Each service is a standalone FastAPI app. Launch from the service directory:

```bash
cd services/<service-name>
uvicorn app.main:app --reload --port <port>
```

Install dependencies from the root `requirements.txt`. Services use `.env` files for configuration — copy and fill in the required variables before running.

## ML Training Pipeline

All scripts live in `train/`. Dataset config: `train/config/data.yaml` (single class: ship).

```bash
# Train YOLO model
python train/train.py --data train/config/data.yaml --model yolo11n.pt --epochs 100

# Run inference on a directory of images
python train/predict.py --weights runs/train/weights/best.pt --source /path/to/images --output /path/to/output

# Evaluate on val or test split
python train/evaluate.py --weights runs/train/weights/best.pt --data train/config/data.yaml --split val
```

Training requires CUDA (torch+cu124). Weights are saved to `runs/` by default.

## Architecture

Four independent FastAPI microservices sharing a PostgreSQL+PostGIS database. No API gateway — services communicate only through the shared database (not via HTTP calls to each other).

| Service | Port (conventional) | Responsibility |
|---|---|---|
| `user-service` | — | Auth, users, groups, JWT + refresh tokens |
| `map-service` | — | CRUD on geographic polygons (areas) |
| `image-service` | — | Fetch Sentinel-2 imagery via Sentinel Hub API, cache previews in Redis |
| `analysis-service` | — | Run Faster R-CNN detection on GeoTIFF images, store results in PostGIS |

### User Workflow

1. Create a polygon in **map-service**
2. Request previews from **image-service** (fetches from Sentinel Hub, caches in Redis, returns Base64 PNG)
3. Save images to disk as GeoTIFF
4. Submit saved images to **analysis-service** for detection
5. Retrieve detection results (bounding boxes as PostGIS geometries, EPSG:4326)

### Key Patterns

- All DB access is async (SQLAlchemy 2.0 async + asyncpg)
- Each service initializes its database schema and seed data on startup via `DB_INITIALIZER.init_database()`
- The ML model is loaded once at startup and stored in `app.state` for reuse across requests
- CORS is fully permissive on all services (all origins allowed)
- Image processing pipeline: Sentinel Hub → GeoTIFF → OpenCV → PyTorch tensor → detections → geo-coordinates via rasterio

### Authentication

`user-service` uses `fastapi-users` with JWT access tokens + UUID refresh tokens. Refresh token rotation is tied to a device fingerprint (browser fingerprint stored in `device_fingerprint` table). Password reset uses SMTP email.

### Geospatial Storage

All geometries stored in PostGIS with SRID=4326. Polygon invariants enforced at the service level: minimum 3 unique coordinates, closed, no self-intersections, area within [1, 2000] m², max 100 polygons per user.

### Image Tiling

`image-service` uses Sentinel Hub's `BBoxSplitter` to tile large polygons into ≤800×800 px sub-images before fetching, since the analysis model processes at 800×800 px.

## Environment Variables

| Service | Key Variables |
|---|---|
| `user-service` | `POSTGRES_DSN_ASYNC`, `JWT_SECRET`, `RESET_PASSWORD_TOKEN_SECRET`, `VERIFICATION_TOKEN_SECRET`, `OWN_EMAIL`, `OWN_EMAIL_PASSWORD`, `SMTP_SERVER`, `SMTP_PORT`, `DEFAULT_GROUPS_CONFIG_PATH` |
| `image-service` | `POSTGRES_DSN_ASYNC`, `CLIENT_ID`, `CLIENT_SECRET` (Sentinel Hub), `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` |
| `analysis-service` | `POSTGRES_DSN_ASYNC`, `MODEL_PATH`, `WIDTH_IMAGE`, `HEIGHT_IMAGE`, `DEFAULT_OBJECTS_CONFIG_PATH` |
| `map-service` | `POSTGRES_DSN_ASYNC` |

## Seed Data

- `services/user-service/default-groups.json` — initial groups: "Администратор" (id=1), "Пользователь" (id=2)
- `services/analysis-service/default-objects.json` — initial object type: "Корабль" (id=1)
