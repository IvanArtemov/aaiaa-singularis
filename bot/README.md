# Telegram Bot - PDF to Knowledge Graph

Telegram бот для автоматического анализа научных статей и построения графов знаний.

## 🎯 Функциональность

Бот принимает PDF научной статьи и:
1. ✅ Извлекает текст из PDF
2. ✅ Анализирует статью с помощью LLM (gpt-5-mini)
3. ✅ Извлекает сущности: факты, гипотезы, эксперименты, результаты, выводы
4. ✅ Строит связи между сущностями
5. ✅ Генерирует SVG граф знаний
6. ✅ Отправляет результат пользователю

## 📋 Требования

- Python 3.10+
- Telegram Bot Token (от @BotFather)
- OpenAI API Key

## 🚀 Установка

### 1. Установите зависимости

```bash
pip install -r requirements.txt
```

### 2. Создайте Telegram бота

1. Откройте Telegram и найдите [@BotFather](https://t.me/botfather)
2. Отправьте команду `/newbot`
3. Следуйте инструкциям и получите токен
4. Скопируйте токен (формат: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 3. Настройте переменные окружения

Создайте файл `.env` в корне проекта:

```bash
cp .env.example .env
```

Отредактируйте `.env` и добавьте:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
OPENAI_API_KEY=your_openai_key
```

Опциональные настройки:
```bash
MAX_PDF_SIZE_MB=10                      # Максимальный размер PDF
MAX_REQUESTS_PER_USER_PER_HOUR=5        # Лимит запросов на пользователя
PROCESSING_TIMEOUT_SECONDS=180          # Таймаут обработки
LLM_MODEL=gpt-5-mini                   # Модель LLM
```

## 🏃 Запуск

```bash
python scripts/run_telegram_bot.py
```

Вы увидите:
```
==================================================
TELEGRAM BOT - PDF to Knowledge Graph
==================================================

🤖 Starting Telegram Bot...
🚀 Bot is starting...
   Press Ctrl+C to stop
==================================================

Bot started successfully!
Temp directory: /path/to/AAIAA/bot/temp
Max file size: 10 MB
Rate limit: 5 requests/hour
```

Бот готов к работе! Найдите его в Telegram и отправьте `/start`.

## 💬 Использование

### Команды

- `/start` - Начать работу сботом
- `/help` - Получить справку
- `/stats` - Посмотреть свою статистику

### Отправка PDF

1. Отправьте PDF файл научной статьи боту
2. Дождитесь обработки (30-60 секунд)
3. Получите SVG граф знаний

### Пример взаимодействия

**Пользователь:** `/start`

**Бот:**
```
👋 Привет! Я бот для анализа научных статей.

📄 Отправь мне PDF научной статьи, и я:
✓ Извлеку ключевые сущности (факты, гипотезы, результаты)
✓ Построю граф знаний
✓ Создам красивую SVG визуализацию

⚡ Стоимость: ~$0.03 за статью
⏱ Время обработки: 30-60 секунд
📊 Ограничение: 5 запросов в час

Попробуй! Просто отправь PDF файл.
```

**Пользователь:** *[отправляет PDF]*

**Бот:**
```
📥 Downloading PDF... ✅
📄 Parsing PDF... ✅
   • Pages: 12
   • Words: 5,432
   • Sections: 6

🤖 Extracting knowledge...
   This will take 30-60 seconds ⏳
```

*(через 45 секунд)*

**Бот:** *[отправляет SVG файл]*
```
✅ Extraction Complete!

📊 Statistics:
• Entities: 47
• Relationships: 38
• Processing time: 45.3s
• Cost: $0.0287

📦 Entities by type:
   Facts: 8
   Hypotheses: 4
   Technique: 12
   Result: 15
   Conclusion: 8

💡 Open the SVG file to explore the knowledge graph!
```

## 📊 Ограничения

- **Размер файла:** Максимум 10 MB
- **Rate limit:** 5 запросов в час на пользователя
- **Формат:** Только PDF файлы
- **Таймаут:** 3 минуты на обработку

## 🏗️ Архитектура

```
bot/
├── __init__.py              # Package init
├── config.py                # Configuration management
├── telegram_bot.py          # Main bot application
├── handlers.py              # Message handlers
├── session_manager.py       # Rate limiting & user tracking
├── exceptions.py            # Custom exceptions
├── utils.py                 # Helper functions
└── temp/                    # Temporary files (auto-cleanup)
```

### Workflow

```
User sends PDF
    ↓
Validate file (size, type)
    ↓
Check rate limit
    ↓
Download PDF → Parse PDF → Extract entities → Generate SVG
    ↓
Send SVG to user
    ↓
Cleanup temp files
```

## 🔧 Компоненты

### Session Manager
- Rate limiting (5 запросов/час)
- User statistics tracking
- Cost tracking
- Active session management

### Handlers
- `/start`, `/help`, `/stats` commands
- PDF document processing
- Error handling
- Progress updates

### Utils
- File validation
- Safe cleanup
- Metrics formatting
- Status messages

## 📈 Мониторинг

### Логирование

Бот автоматически логирует:
- User actions (upload, commands)
- Processing steps (parse, extract, generate)
- Errors and exceptions
- Performance metrics

Логи выводятся в консоль в формате:
```
2025-10-11 15:30:45 - bot.handlers - INFO - User 12345 uploaded document: paper.pdf
2025-10-11 15:31:20 - bot.handlers - INFO - Extraction complete: 47 entities, 38 relationships
```

### Статистика пользователя

Команда `/stats` показывает:
- Всего запросов
- Суммарная стоимость
- Извлечено сущностей
- Запросов за последний час
- Дата первого запроса

## 🛡️ Безопасность

### Rate Limiting
- Защита от спама: максимум 5 запросов в час
- Отслеживание активных сессий
- Блокировка одновременных запросов от одного пользователя

### Cleanup
- Автоматическое удаление временных файлов
- Cleanup старых файлов каждый час
- Cleanup старых сессий каждые 7 дней

### Error Handling
- Graceful degradation при ошибках
- User-friendly сообщения об ошибках
- Automatic recovery

## 🚀 Production Deployment

### Опции деплоя

#### 1. Local Server
```bash
# Запуск в screen/tmux
screen -S telegram_bot
python scripts/run_telegram_bot.py
# Ctrl+A, D для detach
```

#### 2. Systemd Service (Linux)
```bash
# Создайте /etc/systemd/system/telegram-bot.service
[Unit]
Description=Telegram Knowledge Graph Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/AAIAA
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python scripts/run_telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target

# Запуск
sudo systemctl start telegram-bot
sudo systemctl enable telegram-bot
```

#### 3. Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "examples/run_telegram_bot.py"]
```

#### 4. Cloud Hosting
- **Heroku:** Free tier
- **Railway.app:** Free tier
- **DigitalOcean:** $5/month droplet
- **AWS EC2:** t2.micro free tier

## 🐛 Troubleshooting

### Bot не отвечает
- Проверьте `TELEGRAM_BOT_TOKEN` в `.env`
- Убедитесь что бот запущен
- Проверьте логи на ошибки

### Ошибка "API key not found"
- Проверьте `OPENAI_API_KEY` в `.env`
- Убедитесь что файл `.env` в корне проекта

### PDF не обрабатывается
- Проверьте размер файла (макс. 10MB)
- Убедитесь что файл - валидный PDF
- Проверьте логи на ошибки парсинга

### Rate limit exceeded
- Подождите 1 час с последнего запроса
- Или увеличьте `MAX_REQUESTS_PER_USER_PER_HOUR` в `.env`

## 📝 Development

### Тестирование
```bash
# Unit tests (TODO)
pytest tests/bot/

# Manual testing
python scripts/run_telegram_bot.py
```

### Добавление новых функций

#### Новая команда
1. Добавьте handler в `bot/handlers.py`
2. Зарегистрируйте в `bot/telegram_bot.py`

#### Новая функциональность
1. Создайте утилиту в `bot/utils.py`
2. Используйте в handlers

## 📚 Дополнительные ресурсы

- [python-telegram-bot документация](https://docs.python-telegram-bot.org/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [BotFather](https://t.me/botfather)

## 🤝 Support

Вопросы и баги: создайте issue в репозитории

---

**Последнее обновление:** 11 октября 2025
