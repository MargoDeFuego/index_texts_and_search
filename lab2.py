import xml.etree.ElementTree as ET
import json
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

# Загрузка служебных слов
def load_service_words(path="service_words.xml"):
    tree = ET.parse(path)
    root = tree.getroot()
    # приведение к нижнему регистру
    return {w.text.lower() for w in root.findall("word")}


# Загрузка senses.xml
def load_senses(path="senses.xml"):
    tree = ET.parse(path)
    root = tree.getroot()
    senses = {}
    for s in root.findall("sense"):
        name = s.attrib["name"].lower()
        lemma = s.attrib["lemma"].lower()
        senses[name] = lemma
    return senses

# Загрузка базы запросов
def load_requests(path="requests.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["requests"]

# Лемматизация одного слова
def lemmatize_word(word):
    return morph.parse(word)[0].normal_form

# Нормализация запроса
def normalize_query(query, service_words):
    words = query.lower().split()
    return [w for w in words if w not in service_words]

# Разбиение на двусловия
def make_bigrams(words):
    if len(words) < 2:
        return []
    return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

# Получение лемм для двусловий
def get_lemmas(bigrams, senses):
    lemmas = []

    for bg in bigrams:
        bg_lower = bg.lower()

        # Полное совпадение
        if bg_lower in senses:
            lemmas.extend(senses[bg_lower].split())
            continue

        # Частичное совпадение
        found = False
        for sense_name in senses:
            if sense_name in bg_lower:
                lemmas.extend(senses[sense_name].split())
                found = True
                break

        if found:
            continue

        # fallback
        w1, w2 = bg_lower.split()
        lemmas.append(lemmatize_word(w1))
        lemmas.append(lemmatize_word(w2))

    return list(set(lemmas))

# Поиск ответа
def find_answer(lemmas, requests):
    for req in requests:
        if set(req["lemmas"]) == set(lemmas):
            return req["answer"]
    return "Ответ не найден"

# Основная функция
def process_query(query):
    service_words = load_service_words()
    senses = load_senses()
    requests = load_requests()

    normalized = normalize_query(query, service_words)
    bigrams = make_bigrams(normalized)
    lemmas = get_lemmas(bigrams, senses)

    return find_answer(lemmas, requests)


# Примеры
if __name__ == "__main__":
    print("Тест 1:", process_query("как попасть в библиотеку"))
    print("Тест 2:", process_query("где купить билет"))
    print("Тест 3:", process_query("как добраться до университета"))
    print("Тест 4:", process_query("где находится столовая"))
    print("Тест 5:", process_query("как пройти к парку"))
    print("Тест 6:", process_query("как записаться на курс"))
    print("Тест 7:", process_query("как получить справку"))
    print("Тест 8:", process_query("где купить еду"))
    print("Тест 9:", process_query("где находится общежитие"))
    print("Тест 10:", process_query("как найти расписание"))



