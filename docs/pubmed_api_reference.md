# PubMed E-utilities API Reference

## Обзор

E-utilities (Entrez Programming Utilities) - это набор из девяти серверных программ, предоставляющих стабильный интерфейс для доступа к системе баз данных NCBI Entrez.

**Base URL:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

**Поддерживаемые базы данных:** 38+ биомедицинских баз данных, включая PubMed, PubMed Central, Gene, Protein и другие.

---

## Девять основных утилит

| Утилита | Назначение |
|---------|------------|
| **EInfo** | Статистика и метаданные баз данных |
| **ESearch** | Текстовый поиск, возвращает список UIDs |
| **EPost** | Загрузка списков UIDs на History Server |
| **ESummary** | Получение кратких сводок о документах |
| **EFetch** | Получение полных записей данных |
| **ELink** | Поиск связанных записей между базами |
| **EGQuery** | Глобальный поиск по всем базам |
| **ESpell** | Проверка орфографии и предложения |
| **ECitMatch** | Поиск цитирований в PubMed |

---

## ESearch - Поиск статей

### Назначение
Поиск в базах данных Entrez и получение списка уникальных идентификаторов (UIDs/PMIDs).

### Endpoint
```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
```

### Обязательные параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| `db` | База данных | `pubmed` |
| `term` | Поисковый запрос | `crispr cas9` |

### Опциональные параметры

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `retmax` | Макс. кол-во результатов | 20 (макс: 10,000) |
| `retstart` | Начальный индекс | 0 |
| `retmode` | Формат ответа | `xml` (также: `json`) |
| `sort` | Сортировка | `relevance` (также: `pub_date`) |
| `usehistory` | Сохранить на History Server | `n` (использовать `y`) |
| `api_key` | API ключ для увеличения лимита | - |

### Примеры запросов

#### Базовый поиск
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=breast+cancer&retmode=json&retmax=10
```

#### Поиск с фильтром "Free Full Text"
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=crispr+cas9+AND+free+full+text[filter]&retmode=json&retmax=20
```

#### Поиск по журналу и дате
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=science[journal]+AND+breast+cancer+AND+2008[pdat]
```

### Пример ответа (JSON)
```json
{
  "esearchresult": {
    "count": "15234",
    "retmax": "20",
    "retstart": "0",
    "idlist": ["37845123", "37842456", "37839871", ...],
    "translationset": [...],
    "querytranslation": "crispr cas9 AND free full text[filter]"
  }
}
```

---

## EFetch - Получение полных записей

### Назначение
Получение форматированных данных для списка UIDs.

### Endpoint
```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
```

### Обязательные параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| `db` | База данных | `pubmed` |
| `id` | Список UIDs (через запятую) | `17284678,9997` |

### Опциональные параметры

| Параметр | Описание | Значения |
|----------|----------|----------|
| `retmode` | Формат ответа | `xml`, `text` |
| `rettype` | Тип записи | `abstract`, `medline`, `full` |
| `retstart` | Начальный индекс | 0 |
| `retmax` | Макс. кол-во записей | 10,000 |

### Примеры запросов

#### Получение абстрактов в XML
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17284678,9997&retmode=xml&rettype=abstract
```

#### Получение абстрактов в текстовом формате
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=11748933&retmode=text&rettype=abstract
```

### Структура XML ответа (PubMed)

Основные теги для парсинга:
```xml
<PubmedArticle>
  <MedlineCitation>
    <PMID>12345678</PMID>
    <Article>
      <ArticleTitle>Название статьи</ArticleTitle>
      <Abstract>
        <AbstractText>Текст абстракта</AbstractText>
      </Abstract>
      <AuthorList>
        <Author>
          <LastName>Иванов</LastName>
          <ForeName>Иван</ForeName>
        </Author>
      </AuthorList>
      <Journal>
        <Title>Название журнала</Title>
      </Journal>
      <PubDate>
        <Year>2024</Year>
        <Month>Jan</Month>
      </PubDate>
      <KeywordList>
        <Keyword>keyword1</Keyword>
      </KeywordList>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pubmed">12345678</ArticleId>
      <ArticleId IdType="doi">10.1234/example</ArticleId>
      <ArticleId IdType="pmc">PMC1234567</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
```

---

## ELink - Поиск связанных записей

### Назначение
Поиск связей между записями в разных базах данных (например, PMID → PMC ID).

### Endpoint
```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi
```

### Параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| `dbfrom` | Исходная БД | `pubmed` |
| `db` | Целевая БД | `pmc` |
| `id` | UID в исходной БД | `12345678` |
| `retmode` | Формат ответа | `json`, `xml` |

### Пример: Получение PMC ID из PMID
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pmc&id=17284678&retmode=json
```

---

## Поисковые фильтры и теги

### Основные фильтры (Field Tags)

| Тег | Описание | Пример |
|-----|----------|--------|
| `[tiab]` | Заголовок/Абстракт | `cancer[tiab]` |
| `[au]` | Автор | `Smith J[au]` |
| `[ta]` или `[journal]` | Журнал | `Nature[ta]` |
| `[dp]` или `[pdat]` | Дата публикации | `2024[pdat]` |
| `[mh]` | MeSH термины | `Diabetes Mellitus[mh]` |
| `[pt]` | Тип публикации | `Review[pt]` |
| `[sb]` или `[filter]` | Фильтры/Подмножества | `free full text[filter]` |

### Важные фильтры доступности текста

| Фильтр | Описание |
|--------|----------|
| `free full text[filter]` | Статьи с бесплатным полным текстом |
| `free full text[sb]` | Альтернативный синтаксис |
| `pubmed pmc[sb]` | Статьи в PubMed Central |
| `open access[filter]` | Open Access статьи |

### Типы публикаций (Publication Types)

- `Clinical Trial[pt]`
- `Review[pt]`
- `Systematic Review[pt]`
- `Meta-Analysis[pt]`
- `Case Reports[pt]`
- `Randomized Controlled Trial[pt]`

### Диапазоны дат

#### Формат: `YYYY/MM/DD[dp]`

**Одна дата:**
```
cancer AND 2024/01/15[dp]
```

**Диапазон дат:**
```
heart disease AND 2019/01/01:2023/12/31[dp]
```

**Последний год:**
```
covid-19 AND ("2023/01/01"[dp] : "2023/12/31"[dp])
```

### Комбинирование фильтров

**Boolean операторы:** `AND`, `OR`, `NOT`

#### Примеры сложных запросов

**Систематические обзоры с Free Full Text:**
```
alzheimer AND systematic review[pt] AND free full text[filter]
```

**Статьи из определенного журнала за период:**
```
crispr AND Nature[journal] AND 2020:2024[pdat]
```

**Clinical trials с Open Access:**
```
diabetes AND clinical trial[pt] AND open access[filter]
```

**Исключение типов публикаций:**
```
cancer NOT review[pt] NOT case report[pt]
```

---

## Rate Limits и Best Practices

### 🚨 Критические ограничения (чтобы не заблокировали)

**Что приведет к БЛОКИРОВКЕ IP или API ключа:**

❌ **Превышение rate limits**
❌ **Отсутствие параметров `tool` и `email`** (ОБЯЗАТЕЛЬНО!)
❌ **Повторные идентичные запросы** без кэширования
❌ **Большие задачи в рабочее время** (9:00-21:00 EST, пн-пт)
❌ **Нарушение политики использования NCBI**

### Лимиты запросов

| Условие | Запросов/секунду | Примечание |
|---------|------------------|------------|
| Без API ключа | 3 | Базовый лимит |
| С API ключом | 10 | **Рекомендуется** |
| По запросу в NCBI | >10 | Нужно согласование |

### Получение API ключа

1. Зарегистрируйтесь на NCBI: https://www.ncbi.nlm.nih.gov/account/
2. Перейдите в Settings → API Key Management
3. Создайте новый ключ
4. **Один ключ на аккаунт** (нельзя создать несколько)

### 🔴 ОБЯЗАТЕЛЬНЫЕ требования

**КРИТИЧНО:** Всегда указывайте параметры `tool` и `email` в каждом запросе!

```python
# ✓ ПРАВИЛЬНО - с tool и email
params = {
    "db": "pubmed",
    "term": "cancer",
    "tool": "MyApp_Fetcher",  # ОБЯЗАТЕЛЬНО!
    "email": "your@email.com", # ОБЯЗАТЕЛЬНО!
    "api_key": "YOUR_KEY"
}

# ✗ НЕПРАВИЛЬНО - без tool/email (приведет к блокировке!)
params = {
    "db": "pubmed",
    "term": "cancer",
    "api_key": "YOUR_KEY"
}
```

### Регистрация tool и email

**Рекомендуется** (но не обязательно) отправить email в NCBI для регистрации вашего приложения:

**Кому:** eutilities@ncbi.nlm.nih.gov
**Тема:** Tool Registration
**Содержание:**
```
Tool name: AAIAA_PubMed_Fetcher
Email: your@email.com
Description: Academic research project for extracting scientific paper metadata
Expected usage: ~1000 requests/day
```

### Расписание для больших задач

| Время (EST) | День | Рекомендация |
|-------------|------|--------------|
| 9:00 - 21:00 | Пн-Пт | ❌ Избегать больших задач |
| 21:00 - 9:00 | Пн-Пт | ✅ Можно |
| Любое время | Сб-Вс | ✅ Идеально для больших задач |

**Большая задача** = более 100 запросов или загрузка >10,000 записей

### Best Practices

✅ **DO:**
- **ВСЕГДА** указывайте `tool` и `email` параметры (критично!)
- Используйте API ключ для увеличения лимита до 10 req/sec
- Кэшируйте результаты локально (избегайте повторных запросов)
- Используйте History Server (`usehistory=y`) для результатов >10,000
- Планируйте большие задачи на выходные или 21:00-5:00 EST
- Добавляйте задержки между запросами (минимум 100ms с API key)
- Обрабатывайте ошибки и retry с exponential backoff

❌ **DON'T:**
- **НЕ запускайте скрипты без `tool` и `email`** (блокировка!)
- Не превышайте rate limits (3 req/sec без ключа, 10 с ключом)
- Не делайте одинаковые запросы повторно
- Не игнорируйте HTTP 429 (Too Many Requests)
- Не используйте множество API ключей (один на аккаунт)

### Как избежать блокировки

**Минимальные требования:**
1. ✅ Указывать `tool` и `email` в КАЖДОМ запросе
2. ✅ Соблюдать rate limits (макс 10 req/sec)
3. ✅ Кэшировать результаты
4. ✅ Обрабатывать HTTP 429 и делать retry

**Признаки возможной блокировки:**
- HTTP 429 (Too Many Requests)
- HTTP 403 (Forbidden)
- Внезапные таймауты
- Сообщения об ошибках от NCBI

**Если вас заблокировали:**
1. Прекратите запросы немедленно
2. Проверьте, что используете `tool` и `email`
3. Подождите 24 часа
4. Напишите в NCBI: eutilities@ncbi.nlm.nih.gov

### Пример с API ключом
```bash
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&api_key=YOUR_API_KEY&tool=my_script&email=user@example.com
```

---

## Использование History Server

### Зачем нужен?
- Работа с большими наборами результатов (>10,000)
- Комбинирование результатов нескольких запросов
- Эффективная пакетная обработка

### Как использовать?

**Шаг 1: Поиск с сохранением на сервер**
```bash
esearch.fcgi?db=pubmed&term=cancer&usehistory=y
```

**Ответ содержит:**
```xml
<WebEnv>MCID_abc123...</WebEnv>
<QueryKey>1</QueryKey>
<Count>150000</Count>
```

**Шаг 2: Получение результатов частями**
```bash
efetch.fcgi?db=pubmed&query_key=1&WebEnv=MCID_abc123&retstart=0&retmax=500
efetch.fcgi?db=pubmed&query_key=1&WebEnv=MCID_abc123&retstart=500&retmax=500
```

---

## Примеры Python кода

### Базовый поиск с ESearch
```python
import requests

def search_pubmed(query, max_results=10):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    return data["esearchresult"]["idlist"]

# Использование
pmids = search_pubmed("crispr cas9 AND free full text[filter]", max_results=20)
print(f"Found {len(pmids)} articles")
```

### Получение метаданных с EFetch
```python
import requests
import xml.etree.ElementTree as ET

def fetch_pubmed_articles(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract"
    }

    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.content)

    articles = []
    for article in root.findall(".//PubmedArticle"):
        title = article.find(".//ArticleTitle").text
        pmid = article.find(".//PMID").text
        articles.append({"pmid": pmid, "title": title})

    return articles
```

### Поиск с фильтром Free Full Text
```python
def search_free_full_text(query, max_results=20):
    # Автоматически добавляем фильтр
    query_with_filter = f"{query} AND free full text[filter]"

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query_with_filter,
        "retmode": "json",
        "retmax": max_results
    }

    response = requests.get(base_url, params=params)
    return response.json()["esearchresult"]["idlist"]

# Примеры использования
pmids = search_free_full_text("crispr cas9")
pmids = search_free_full_text("alzheimer AND review[pt]")
```

---

## Ссылки на официальную документацию

- **E-utilities Overview:** https://www.ncbi.nlm.nih.gov/books/NBK25497/
- **ESearch Documentation:** https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
- **EFetch Documentation:** https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch
- **Search Field Descriptions:** https://pubmed.ncbi.nlm.nih.gov/help/
- **API Key Registration:** https://www.ncbi.nlm.nih.gov/account/settings/

---

## Часто используемые комбинации запросов

### Поиск недавних обзоров с Free Full Text
```
{query} AND review[pt] AND ("2023"[pdat] : "3000"[pdat]) AND free full text[filter]
```

### Поиск RCT (Randomized Controlled Trials)
```
{query} AND randomized controlled trial[pt] AND free full text[filter]
```

### Поиск Meta-анализов
```
{query} AND meta-analysis[pt] AND free full text[filter]
```

### Поиск по конкретному автору
```
{query} AND Smith J[au] AND free full text[filter]
```

---

**Дата создания:** 2025-10-08
**Версия:** 1.0
