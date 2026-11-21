import re
import numpy as np
import PyPDF2
import nltk
import gensim.downloader as api
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# ============================================================
# 1. Чтение PDF с произведениями М. Ю. Лермонтова
# ============================================================

# Положи PDF в папку с ноутбуком или укажи полный путь к файлу.
# Например: Lermontov.pdf

def clean_pdf_text(text):
    # Удаление переносов слов на границах строк
    text = re.sub(r'-\s*\n\s*', '', text)

    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\r', ' ')

    return text.strip()


def read_pdf(path):
    text = ""

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return clean_pdf_text(text)


text = read_pdf("lermontov-izbrannoe-1983.pdf")

print("Длина текста:", len(text))
print("Начало текста:", text[:500])


# ============================================================
# 2. Предобработка текста
# ============================================================

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab", quiet=True)

russian_stopwords = set(stopwords.words("russian"))


def preprocess_text(text):
    tokens = word_tokenize(text.lower(), language="russian")

    # Оставляем только слова из русских букв
    tokens = [
        word for word in tokens
        if re.fullmatch(r"[а-яё]+", word)
    ]

    # Удаляем стоп-слова
    tokens = [
        word for word in tokens
        if word not in russian_stopwords
    ]

    return tokens


# Разбиваем текст на предложения
sentences = nltk.sent_tokenize(text, language="russian")

# Токенизируем каждое предложение отдельно
tokenized_sentences = [
    preprocess_text(sentence)
    for sentence in sentences
]

# Убираем пустые предложения
tokenized_sentences = [
    sentence for sentence in tokenized_sentences
    if sentence
]

print("Количество предложений:", len(tokenized_sentences))

if tokenized_sentences:
    print("Пример токенов:", tokenized_sentences[0])


# ============================================================
# 3. Обучение моделей Word2Vec
# ============================================================

# Skip-gram
model_sg = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    seed=42,
    epochs=100
)

# CBOW
model_cbow = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    workers=4,
    seed=42,
    epochs=100
)

print("Модели обучены.")
print("Размер словаря Skip-gram:", len(model_sg.wv))
print("Размер словаря CBOW:", len(model_cbow.wv))


# ============================================================
# 4. Предобученная модель Google News
# ============================================================

# Англоязычная модель для сравнения.
# Скачивается при первом запуске и требует интернет.

pretrained_model = api.load("word2vec-google-news-300")

print("Предобученная модель Google News загружена.")


# ============================================================
# 5. Поиск похожих слов
# ============================================================

# Ключевые слова, связанные с произведениями Лермонтова
keywords = [
    "парус",
    "море",
    "буря",
    "туча",
    "одиночество",
    "свобода",
    "любовь",
    "душа"
]

for word in keywords:
    print(f"\nСлово: {word}")

    if word in model_sg.wv:
        print(
            "Skip-gram:",
            model_sg.wv.most_similar(word, topn=5)
        )
    else:
        print("Skip-gram: слова нет в модели")

    if word in model_cbow.wv:
        print(
            "CBOW:",
            model_cbow.wv.most_similar(word, topn=5)
        )
    else:
        print("CBOW: слова нет в модели")


# ============================================================
# 6. Аналогии
# ============================================================

# Аналогия для русских моделей:
# море + парус - буря
# Это экспериментальный пример, результат не гарантирован.

analogy_positive = ["море", "парус"]
analogy_negative = ["буря"]


def run_analogy(model, positive, negative, model_name):
    all_words = positive + negative

    if all(word in model.wv for word in all_words):
        result = model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=5
        )

        print(f"{model_name}:")
        print(result)
    else:
        missing = [
            word for word in all_words
            if word not in model.wv
        ]

        print(f"{model_name}: отсутствуют слова {missing}")


run_analogy(
    model_sg,
    analogy_positive,
    analogy_negative,
    "Skip-gram (Лермонтов)"
)

run_analogy(
    model_cbow,
    analogy_positive,
    analogy_negative,
    "CBOW (Лермонтов)"
)


# Аналогия в английской модели Google News
if all(
    word in pretrained_model
    for word in ["sea", "sail", "storm"]
):
    result_gn = pretrained_model.most_similar(
        positive=["sea", "sail"],
        negative=["storm"],
        topn=5
    )

    print("Google News:", result_gn)
else:
    print("Google News: не все слова есть в модели")


# ============================================================
# 7. Две темы: природа + чувства
# ============================================================

# Русские слова для моделей, обученных на Лермонтове

topic_nature_ru = [
    "горы", "море", "небо", "земля", "лес",
    "скала", "туча", "буря", "ветер", "звезда"
]

topic_feelings_ru = [
    "любовь", "печаль", "грусть", "одиночество", "тоска",
    "свобода", "надежда", "страдание", "мечта", "душа"
]

words_ru = topic_nature_ru + topic_feelings_ru


# Английские аналоги для Google News

topic_nature_en = [
    "mountain", "sea", "sky", "earth", "forest",
    "rock", "cloud", "storm", "wind", "star"
]

topic_feelings_en = [
    "love", "sadness", "sorrow", "loneliness", "longing",
    "freedom", "hope", "suffering", "dream", "soul"
]

words_en = topic_nature_en + topic_feelings_en


# ============================================================
# 8. Фильтрация слов для каждой модели
# ============================================================

def filter_words(model, words):
    return [word for word in words if word in model.wv]


filtered_sg = filter_words(model_sg, words_ru)
filtered_cbow = filter_words(model_cbow, words_ru)

filtered_pretrained = [
    word for word in words_en
    if word in pretrained_model
]

print("\nСлова для визуализации:")
print("Skip-gram:", len(filtered_sg), "из", len(words_ru))
print("CBOW:", len(filtered_cbow), "из", len(words_ru))
print(
    "Google News:",
    len(filtered_pretrained),
    "из",
    len(words_en)
)


# ============================================================
# 9. Визуализация векторов с помощью t-SNE
# ============================================================

def plot_tsne(model_vectors, labels, title, colors=None):
    if len(model_vectors) < 3:
        print(
            f"{title}: недостаточно слов для t-SNE "
            f"({len(model_vectors)})."
        )
        return

    vectors = np.array(model_vectors)

    perplexity_value = min(10, len(model_vectors) - 1)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity_value,
        init="pca",
        learning_rate="auto"
    )

    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))

    if colors is None:
        colors = ["royalblue"] * len(labels)

    for i, word in enumerate(labels):
        plt.scatter(
            reduced[i, 0],
            reduced[i, 1],
            color=colors[i],
            s=60
        )

        plt.annotate(
            word,
            (reduced[i, 0], reduced[i, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# Цвета для двух тематических групп:
# синий — природа, красный — чувства

def get_word_colors(words, nature_words, feelings_words):
    colors = []

    for word in words:
        if word in nature_words:
            colors.append("royalblue")
        elif word in feelings_words:
            colors.append("tomato")
        else:
            colors.append("gray")

    return colors


# Skip-gram
sg_vectors = [
    model_sg.wv[word]
    for word in filtered_sg
]

plot_tsne(
    sg_vectors,
    filtered_sg,
    "Лермонтов: Skip-gram — природа и чувства",
    get_word_colors(
        filtered_sg,
        topic_nature_ru,
        topic_feelings_ru
    )
)


# CBOW
cbow_vectors = [
    model_cbow.wv[word]
    for word in filtered_cbow
]

plot_tsne(
    cbow_vectors,
    filtered_cbow,
    "Лермонтов: CBOW — природа и чувства",
    get_word_colors(
        filtered_cbow,
        topic_nature_ru,
        topic_feelings_ru
    )
)


# Google News (английские слова)
pretrained_vectors = [
    pretrained_model[word]
    for word in filtered_pretrained
]

plot_tsne(
    pretrained_vectors,
    filtered_pretrained,
    "Google News: nature and feelings",
    get_word_colors(
        filtered_pretrained,
        topic_nature_en,
        topic_feelings_en
    )
)