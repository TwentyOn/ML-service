# ML-service
Приложения для проведения экспериментов с классификации текста спам/не спам
# requirements
Для установки необходимых для работы библиотек выполните:
pip install -r requirements.txt
# Тесты 
Тесты можно запустить следующей командой:
pytest tests/test_ml.py
# Для запуска web-приложения с помощью FAST-API:
uvicorn app.app:app --host 0.0.0.0 --port 8080
