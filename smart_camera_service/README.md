# Smart Camera Price Service

Мини-сервис на FastAPI + Jinja + Bootstrap для тестирования будущей модели оценки цены объекта по фото.

## Запуск

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

После запуска откройте `http://127.0.0.1:8000`.

## Где менять заглушку на настоящую модель

Сейчас ответ формируется в `lib/fake_model.py`.
Будущая модель должна сохранить тот же интерфейс:

```python
prediction = model.predict(image)
```

Где `prediction` содержит:

- `point_price`: точечная оценка цены;
- `min_price`, `max_price`: диапазон цены;
- `microcategory`: предсказанная/подтверждённая микрокатегория;
- `confidence`: условная уверенность;
- `similar_ads`: список похожих объявлений без фиксированной длины.

Каждый похожий объект содержит:

- `image`: PIL Image;
- `price`: цена;
- `title`: название объявления.
