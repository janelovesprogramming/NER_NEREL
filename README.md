# NER Task for Russian Texts

Для обработки содержащихся в задании РУ текстов реализовать решение, которое:

  1) получает на вход текст на русском языке
  2) (1 задача) выделяет словосочетания, посредством функционала аналогичного noun_chunks , и назначает им тег NP
  3) (2 задача) выполняет NER по текстам, в том числе на тег PROFESSION (для этого - натюнить модель, используя open source наборы) при обучении модели стремимся к высокому Precision по тэгу PROFESSION (подумать, как это сделать через кастомную лосс-функцию) по возможности модель должна работать без/с наименьшим использованием GPU
  4) объединяет два результата выше, заменяя тег NP для спана, если он пересекается со спанами от NER-модели (обосновать логику объединения)
  5) выдает на выходе для текста набор терминов с тэгами NP, PROFESSION и др.

Воспроизводимый код обучения и использования выложить на гитхаб, прислать результаты для переданных текстов.

CUDA available: True
Device name: NVIDIA GeForce RTX 3090

# 1. Подготовка данных
Для 2 задачи был выбран набор данных с рускоязычными текстами NEREL. NEREL - набор данных с вложенными именованными сущностями, отношениями, событиями и связями.

Для конвертации данных NEREL в json формат необходимо запустить скрипт `nerel_to_json.py`.

`python3 nerel_to_json.py --dataset_path dataset/NEREL --tags_path dataset/nerel.tags --output_path dataset/nerel_data`

# 2. Environment Setup
`pip install -r /path/to/requirements.txt`

# 3. Training
`python3 src/train.py`

Дисбаланс данных представляет собой серьезную проблему в различных задачах, в частности и в задаче NER. В NER наблюдается дисбаланс данных с распределением в виде длинного хвоста, в котором присутствуют многочисленные классы меньшинства (т.е. классы сущностей) и один класс большинства (т.е, 0-класс). Дисбаланс приводит к ошибочным классификациям классов сущностей как 0-класс. В качестве loss функции была выбрана CrossEntropyLoss с weights для каждого класса сущности, чтобы указать разный вес для классов сущностей, таких как '0' и 'PROFESSION'.

<img width="1065" alt="Screenshot 2024-04-03 at 22 37 34" src="https://github.com/janelovesprogramming/NER_NEREL/assets/35342454/215cb019-d745-4227-9ebc-3932a17116f9">

# 4. Evaluation

```json{
      "eval/RELIGION_f1":0.6,
      "eval/PERSON_f1":0.86886,
      "eval/AGE_f1":0.94382,
      "eval/CITY_f1":0.73068,
      "eval/LOCATION_f1":0.58824,
      "eval/WORK_OF_ART_f1":0.57692,
      "eval/COUNTRY_f1":0.76588,
      "eval/PROFESSION_f1":0.71933,
      "eval/AWARD_f1":0.53061,
      "eval/0__f1":0.81661,
      "eval/loss":0.31946,
      "eval/overall_accuracy":0.95933,
      "eval/overall_f1":0.8005,
      "eval/overall_precision"0.78514,
      "eval/overall_recall":0.81646,
      "test/RELIGION_f1":0.91667,
      "test/PERSON_f1":0.84278,
      "test/AGE_f1":0.90769,
      "test/CITY_f1":0.76674,
      "test/LOCATION_f1":0.42105,
      "test/WORK_OF_ART_f1":0.40602,
      "test/COUNTRY_f1":0.82338,
      "test/PROFESSION_f1":0.67667,
      "test/AWARD_f1":0.67797,
      "test/0__f1":0.77624,
      "test/loss":0.34089,
      "test/overall_accuracy":0.95335,
      "test/overall_f1":0.7744,
      "test/overall_precision":0.75617,
      "test/overall_recall":0.79353,
      "test/runtime":1.3513,
      "train/epoch":5.0,
      "train/grad_norm":1.4194,
      "train/learning_rate":1e-05,
      "train/loss":0.177,
      "train_loss":0.3063,
}
```

# 5. Predict
`python3 src/predict_entities.py`

# Dev set, 10% of all data
```json
{
        "learning_rate": 5e-5,
        "architecture": "BERT",
        "dataset": "NEREL-v1",
        "epochs": 5,
}
