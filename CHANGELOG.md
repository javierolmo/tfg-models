# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 2026-09-30

### Added
- Migración a estructura canónica `src/` y empaquetado con `pyproject.toml`.
- Modelo `LightGBMTrainer` con soporte nativo de variables categóricas.
- CLI unificado (`tfg-models`) y ejecutable `python -m tfg_models`.
- Patrón Template Method en `BaseModelTrainer` para estandarizar el ciclo ML.
- Dockerización completa con `Dockerfile` y servicios en `docker-compose.yml`.
- Flujos de GitHub Actions: CI para PRs y CD para auto-tagging y publicación en Docker Hub.
- Suite de tests unitarios automatizados con 89% de cobertura de código.

### Changed
- Persistencia en producción delegada exclusivamente a `AzureModelHandler`.
- Limpieza y preprocesamiento centralizados en `data/preprocessing.py`.

### Removed
- Eliminado todo el código y wrappers obsoletos fuera del directorio `src/`.
- Extraído `LocalModelHandler` fuera de producción hacia fixtures de test.
