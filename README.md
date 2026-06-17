# комАААнда

## Выбранный проект: Умная камера (15)

Состав команды:

1. Ярослав Волков (капитан, буду загружать решения)
2. Родион Макарьин
3. Даниил Кожин



Оценка проекта до взятия в работу находится в файле Analysis\_and\_evaluation.md

## Запуск сервиса

Сервис находится в папке complete_service.

Артефакты модели скачать можно из Google Drive:

https://drive.google.com/drive/folders/1_3MM0XozNX6TbeODRFKXGKO_sPVI2KSh?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

Пароль указан в презентации.

После скачивания поместите артефакты в папку `artifacts/` в корне проекта:

    artifacts/
    ├── listing_images/
    ├── siglip_train.index
    ├── train_listings_meta.parquet
    ├── v8tuned_meta.json
    └── v8tuned_weights.pt

Сборка и запуск контейнера:

    docker compose build
    docker compose up

После запуска сервис будет доступен по адресу:

    http://localhost:8000

Проверка healthcheck:

    curl http://localhost:8000/health
