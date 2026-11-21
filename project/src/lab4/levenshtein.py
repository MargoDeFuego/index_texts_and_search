def levenshtein(a: str, b: str) -> int:
    """Классическое расстояние Левенштейна: вставка/удаление/замена = 1.

    Перестановка соседних символов НЕ является одной операцией: это намеренно,
    чтобы показать ограничение базового алгоритма из задания.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j]
        for i, ca in enumerate(a, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                cur[i - 1] + 1,      # вставка
                prev[i] + 1,         # удаление
                prev[i - 1] + cost,  # замена
            ))
        prev = cur
    return prev[-1]
