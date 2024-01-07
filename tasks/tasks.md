# Извлечение данных из коллекции новостных текстов

Данные расположены  [в файле data/news.tar.gz](data/news.tar.gz). С некоторых новостных сайтов был загружен архив новостей а  несколько лет, причем каждая новость принаделжит к какой-то рубрике: `science`, `style`, `culture`, `life`, `economics`, `business`, `travel`, `forces`, `media`, `sport`
    

В каждой строке файла содержится метка рубрики, заголовок новостной статьи и сам текст статьи, например:

        sport <tab> Сборная Канады по хоккею разгромила чехов <tab> Сборная Канады по хоккею крупно об...

С помощью [Yargy](https://github.com/natasha/yargy) или [Томита-парсера](https://github.com/yandex/tomita-parser) извлеките данные, которые можно описать структурой вида:


```python
@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]
```

# Контекстно-независимое векторное представление слов 

1. Используйте тексты из предыдущего задания и обучите на них модель векторного представления слов (опционально можно приводить слова к нормальной форме и удалить стоп-слова). Можно использовать `gensim`.

2. Разделите коллекцию текстов на обучающее и тестовое множество. С помощью обученной модели векторного представления отобразите каждый документ в вектор, усреднив все вектора для слов документа. 

3. Используйте какой-либо алгоритм классификации (например `SVM`) для классификации текстов. Для обучения используйте тестовое множество, для анализа результатов - тестовое.

4. Простое усреднение векторов слов обычно не дает хорошего отображения документа в вектор. Придумайте альтернативный способ. Протестируйте его, повторно обучив алгоритм классификации на тех же данных. 

# Применение инструментов Hugging face и предобученных моделей

### Вариант 1:
 Вам нужно создать искусственные данные для тестирования и/или обучения чат-бота. По заданному предложению/утверждению/команде создать набор расширенных предложений/утверждений/команд с приблизительно тем же смыслом. Пример:

> After your workout, remember to focus on maintaining a good water balance.

похожие команды:

> Remember to drink enough water to restore and maintain your body's hydration after your cardio training.

>Please don't forget to maintain water balance after workout.

Предлагается решить упрощенную версию данной задачи с применением общедоступных "маленьких". 
В репозитории Hugging Face есть большое количество предобученных моделей для [casual](https://huggingface.co/models?pipeline_tag=text-generation) и [masked](https://huggingface.co/models?pipeline_tag=fill-mask) языкового моделирования.  Также для валидации можно использовать [sentence-transformers](https://huggingface.co/sentence-transformers). Выбрать нужно модели, которые можно запускать на CPU.

Пример использования masked LM:

```python
import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer

# загружается токенайзер
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
# загружается модель
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

# предложение и замаскированным токеном
sequence = f"My name is {tokenizer.mask_token}."

# результат токенизации
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# применение модели
result = model(input_ids=input_ids)

# индекс замаскированного токена (NB может не совпадать с номером слова)
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# самый вероятный токен 
print(tokenizer.decode(result.logits[:, mask_token_index].argmax()))
```

или через [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="distilroberta-base")

pipe("My name is <mask>.")
```

Casual LM через pipeline:

```python
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')

generator("Hello", max_length=10, num_return_sequences=5)
```

Один наивных способов решения задачи без дополнительного обучения - замаскировать, или вставить в исходную команду замаскированный токен, или обрезать часть команды и применить языковую модель. Результат можно валидировать с помощью [sentence-transformers](https://huggingface.co/sentence-transformers). 

### Вариант 2:

Нужно реализовать простейшую семантическую поисковую систему помощью векторного представления предложений/текстов.
1. Выбрать коллекцию текстовых документов (небольшое подмножество статей из Википедии (из дампа), новости, и т.п.).
2. Выбрать модель для получения векторных представлений (например [sentence-transformers](https://huggingface.co/sentence-transformers)).
3. Выбрать векторное хранилище (faiss, lancedb, qdrant, chroma, pgvector, redis и т.д.)
4. Реализовать поиск, (возможно с постфильтрацией) и продемонстрировать его работу. Индексация и поиск должны быть реализованы в виде отдельных скриптов с CLI.

Нельзя использовать LangChain. 
