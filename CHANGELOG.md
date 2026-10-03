# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-10-03
- Separación en imágenes mínimas para inferencia y entrenamiento con APIs FastAPI dedicadas.
- Autenticación sin secretos con Azure Managed Identity y adaptación al estándar de la capa Gold.

## [0.0.2] - 2026-09-30
- Migración a PyArrow Parquet nativo, eliminación de PySpark y Java 17, reduciendo la imagen Docker a 948MB.

## [0.0.1] - 2026-09-30
- Estructura canónica src/, empaquetado, LightGBM, patrón Template Method, CLI y flujos CI/CD con Docker.
