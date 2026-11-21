import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from lab4.io_utils import load_json, save_json
from lab4.ner_model import OfflineNERModel
from lab4.pipeline import process
from lab4.evaluation import run_combination, baseline_ner
from lab4.experiments import run_dev_experiments
from lab4.levenshtein import levenshtein


def load_resources():
    try:
        word_dict = load_json(ROOT/'data'/'word_dictionary.json')
        display = load_json(ROOT/'data'/'word_display.json')
        entity_dict = load_json(ROOT/'data'/'entity_dictionary.json')
        model = OfflineNERModel.load(ROOT/'models'/'rutiny_ner_v1.json')
        return word_dict, display, entity_dict, model
    except FileNotFoundError as e:
        raise RuntimeError(f'Не найден обязательный локальный файл: {e.filename}. Ничего скачивать не нужно; проверьте, что архив распакован полностью.') from e


def print_result(out):
    print('Исходный текст:', out['original_text'])
    print('Предлагаемый текст:', out['proposed_text'])
    if out['changes']:
        print('Изменения:')
        for c in out['changes']:
            alts=', '.join(f"{a['word']} (d={a['distance']}, f={a['frequency']})" for a in c.get('alternatives',[]))
            print(f"  [{c['start']}, {c['end']}) {c['wrong']!r} -> {c['correct']!r}; d={c['distance']}; варианты: {alts}")
    else:
        print('Изменения: нет')
    if out.get('entity_recommendations'):
        print('Рекомендации внутри защищённых сущностей (не применяются автоматически):')
        for r in out['entity_recommendations']:
            print(' ', r)
    if out['entities']:
        print('Сущности (позиции в предлагаемом тексте):')
        for e in out['entities']:
            print(f"  {e['text']!r}: {e['type']} [{e['start']}, {e['end']}) source={e.get('source','')}")
    else:
        print('Сущности: не найдены')


def cmd_query(args):
    word_dict, display, entity_dict, model = load_resources()
    out = process(args.text, args.recognizer, args.order, word_dict, display, entity_dict, model,
                  threshold=args.threshold, use_frequencies=not args.no_frequencies,
                  protect_entities=not args.no_protection)
    print_result(out)
    if args.output:
        save_json(args.output, out)
        print('Результат сохранён:', args.output)


def cmd_file(args):
    word_dict, display, entity_dict, model = load_resources()
    try:
        records = load_json(args.input)
    except Exception as e:
        print(f'Ошибка загрузки файла: {e}', file=sys.stderr); return 2
    if not isinstance(records, list):
        print('Ожидался JSON-массив запросов.', file=sys.stderr); return 2
    outputs=[]
    for item in records:
        text = item.get('original_text', item.get('text','')) if isinstance(item,dict) else str(item)
        outputs.append(process(text, args.recognizer, args.order, word_dict, display, entity_dict, model,
                               threshold=args.threshold, use_frequencies=not args.no_frequencies,
                               protect_entities=not args.no_protection))
    save_json(args.output, outputs)
    print(f'Обработано запросов: {len(outputs)}. Результат: {args.output}')


def round_metrics(x):
    if isinstance(x,dict): return {k:round_metrics(v) for k,v in x.items()}
    if isinstance(x,list): return [round_metrics(v) for v in x]
    if isinstance(x,float): return round(x,4)
    return x


def evaluate_all(write=True):
    word_dict, display, entity_dict, model = load_resources()
    dev=load_json(ROOT/'data'/'queries_dev.json')
    test=load_json(ROOT/'data'/'queries_test.json')

    dev_exp = run_dev_experiments(dev, word_dict, display, entity_dict, model)
    # Выбор параметров только по dev: более высокая top-1 accuracy, затем меньший false-change rate.
    th_scores=dev_exp['distance_threshold']
    best_th=max((1,2), key=lambda th:(th_scores[str(th)]['typo_accuracy'], -th_scores[str(th)]['false_change_rate'], -th))
    fr_scores=dev_exp['frequency_ranking']
    use_freq = (fr_scores['with_frequency']['typo_accuracy'], -fr_scores['with_frequency']['false_change_rate']) >= (fr_scores['without_frequency']['typo_accuracy'], -fr_scores['without_frequency']['false_change_rate'])
    config={'threshold':best_th,'use_frequencies':use_freq,'normalization':'lowercase + ё→е; hyphens kept; stopwords kept','selected_on':'queries_dev.json only'}

    summary={'config':config,'baseline':{},'combinations':{}}
    details={}
    for recognizer in ('rule','model'):
        b=baseline_ner(test, recognizer, entity_dict, model)
        summary['baseline'][recognizer]=b['metrics']
    for recognizer in ('rule','model'):
        for order in ('A','B'):
            key=f'{recognizer}_{order}'
            combo=run_combination(test, recognizer, order, word_dict, display, entity_dict, model,
                                  threshold=best_th,use_frequencies=use_freq,protect_entities=True)
            summary['combinations'][key]={'correction':combo['correction'],'ner':combo['ner']}
            details[key]=combo['outputs']
    if write:
        save_json(ROOT/'results'/'dev_experiments.json', round_metrics(dev_exp))
        save_json(ROOT/'config'/'final_config.json', config)
        save_json(ROOT/'results'/'test_summary.json', round_metrics(summary))
        save_json(ROOT/'results'/'test_details.json', details)
        # CSV comparison required by report
        with open(ROOT/'results'/'comparison.csv','w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['method','order','typo_accuracy','false_change_rate','precision','recall','f1'])
            for key,v in summary['combinations'].items():
                method,order=key.rsplit('_',1); n=v['ner']['overall']; c=v['correction']
                w.writerow([method,order,round(c['typo_accuracy'],4),round(c['false_change_rate'],4),round(n['precision'],4),round(n['recall'],4),round(n['f1'],4)])
    return round_metrics(summary), round_metrics(dev_exp)


def cmd_evaluate(args):
    summary,_=evaluate_all(write=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nФайлы результатов сохранены в results/.')


def cmd_distance(args):
    print(levenshtein(args.a,args.b))


def build_parser():
    p=argparse.ArgumentParser(description='ЛР4: исправление опечаток и NER, полностью offline')
    sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('query',help='обработать один запрос')
    q.add_argument('text'); q.add_argument('--recognizer',choices=['rule','model'],default='rule')
    q.add_argument('--order',choices=['A','B'],default='A'); q.add_argument('--threshold',type=int,choices=[1,2],default=1)
    q.add_argument('--no-frequencies',action='store_true'); q.add_argument('--no-protection',action='store_true')
    q.add_argument('--output'); q.set_defaults(func=cmd_query)
    f=sub.add_parser('file',help='обработать JSON-файл')
    f.add_argument('input'); f.add_argument('--output',default=str(ROOT/'results'/'file_output.json'))
    f.add_argument('--recognizer',choices=['rule','model'],default='rule'); f.add_argument('--order',choices=['A','B'],default='A')
    f.add_argument('--threshold',type=int,choices=[1,2],default=1); f.add_argument('--no-frequencies',action='store_true'); f.add_argument('--no-protection',action='store_true')
    f.set_defaults(func=cmd_file)
    e=sub.add_parser('evaluate',help='запустить dev-эксперименты и итоговую оценку test'); e.set_defaults(func=cmd_evaluate)
    d=sub.add_parser('distance',help='расстояние Левенштейна'); d.add_argument('a'); d.add_argument('b'); d.set_defaults(func=cmd_distance)
    return p

if __name__=='__main__':
    parser=build_parser(); args=parser.parse_args(); rc=args.func(args); sys.exit(rc or 0)
