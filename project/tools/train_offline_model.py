"""Train and serialize the offline NER model shipped with the project.

The main laboratory program does NOT call this script. The weights are already
stored in models/rutiny_ner_v1.json. This script exists only for reproducibility.
No internet and no third-party packages are used.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lab4.tokenizer import tokenize
from lab4.ner_model import token_features, LABELS


def make_examples(catalog):
    examples = []
    templates = {
        'PERSON': [
            'Биография {e}.', 'Статьи о {e}.', 'Работы {e} опубликованы.',
            'В архиве есть материалы о {e}.', '{e} известен в истории.',
            'Профессор {e} выступил с докладом.'
        ],
        'ORG': [
            'Новости {e}.', 'Документы {e} опубликованы.', 'История {e}.',
            'Материалы подготовлены в {e}.', 'Компания {e} представила проект.',
            'Сотрудники {e} опубликовали отчёт.'
        ],
        'LOC': [
            'История {e}.', 'Новости из {e}.', 'Мы приехали в {e}.',
            'Материалы о {e}.', 'Город {e} упоминается в справочнике.',
            'Экспедиция работала рядом с {e}.'
        ]
    }
    # независимый от queries_test каталог корпуса: формы берутся из entity_catalog
    for item in catalog:
        typ = item['type']
        for form in item.get('forms', []):
            for tpl in templates[typ]:
                text = tpl.format(e=form)
                start = text.index(form); end = start + len(form)
                toks = tokenize(text)
                labels = [typ if t.start >= start and t.end <= end else 'O' for t in toks]
                examples.append((text, labels))
    negatives = [
        'Сегодня опубликованы научные статьи.',
        'Последние новости появились утром.',
        'История науки содержит много открытий.',
        'Показать информацию о музейной коллекции.',
        'Найти известные книги и рассказы.',
        'Где находится музей и как туда доехать?',
        'Научные центры публикуют результаты исследований.',
        'Компания представила новый проект.',
        'Город развивается рядом с крупной рекой.',
        'Профессор выступил перед студентами.',
        'Завод выпустил новые модели автомобилей.',
        'Университет открыл новую лабораторию.'
    ] * 8
    for text in negatives:
        examples.append((text, ['O'] * len(tokenize(text))))
    return examples


class AveragedPerceptron:
    def __init__(self):
        self.weights = {lab: defaultdict(float) for lab in LABELS}
        self.totals = defaultdict(float)
        self.timestamps = defaultdict(int)
        self.step = 0

    def score(self, feats, lab):
        w = self.weights[lab]
        return sum(w.get(f, 0.0) for f in feats)

    def predict(self, feats):
        return max(LABELS, key=lambda lab: (self.score(feats, lab), -LABELS.index(lab)))

    def _update_feat(self, lab, feat, value):
        key=(lab, feat)
        elapsed = self.step - self.timestamps[key]
        self.totals[key] += elapsed * self.weights[lab][feat]
        self.timestamps[key] = self.step
        self.weights[lab][feat] += value

    def update(self, truth, guess, feats):
        self.step += 1
        if truth == guess:
            return
        for feat in feats:
            self._update_feat(truth, feat, 1.0)
            self._update_feat(guess, feat, -1.0)

    def average(self):
        if not self.step:
            return
        for lab in LABELS:
            for feat in list(self.weights[lab]):
                key=(lab, feat)
                total = self.totals[key] + (self.step - self.timestamps[key]) * self.weights[lab][feat]
                avg = total / self.step
                if abs(avg) < 1e-9:
                    del self.weights[lab][feat]
                else:
                    self.weights[lab][feat] = round(avg, 6)


def train(catalog, epochs=12, seed=42):
    examples = make_examples(catalog)
    rng = random.Random(seed)
    model = AveragedPerceptron()
    for epoch in range(epochs):
        rng.shuffle(examples)
        mistakes=0; total=0
        for text, labels in examples:
            toks=tokenize(text)
            for i, truth in enumerate(labels):
                feats=token_features(toks,i)
                guess=model.predict(feats)
                if guess!=truth: mistakes+=1
                model.update(truth,guess,feats)
                total+=1
        print(f'epoch {epoch+1}: errors={mistakes}/{total}')
    model.average()
    return model, len(examples)


def main():
    catalog=json.loads((ROOT/'data'/'entity_catalog.json').read_text(encoding='utf-8'))
    model,n=train(catalog)
    data={
      'metadata':{
        'name':'RuTinyNER-Perceptron',
        'version':'1.0.0',
        'language':'ru',
        'labels':['PERSON','ORG','LOC','O'],
        'architecture':'averaged perceptron token classifier with contextual lexical/shape features',
        'training_source':'data/entity_catalog.json + deterministic Russian templates in tools/train_offline_model.py',
        'training_examples':n,
        'runtime_dependencies':'Python standard library only',
        'note':'Model is pre-trained and shipped ready; training is not executed by the main program.'
      },
      'weights':{lab:dict(w) for lab,w in model.weights.items()}
    }
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'models'/'rutiny_ner_v1.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved model')

if __name__=='__main__': main()
