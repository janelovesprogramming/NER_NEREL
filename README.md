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
Необходимо конвертировать данные NEREL в json формат.

`python3 nerel_to_json.py --dataset_path dataset/NEREL --tags_path dataset/nerel.tags --output_path dataset/nerel_data`

# 2. Environment Setup
`pip install -r /path/to/requirements.txt`

# 3. Training
`python3 src/train.py`

# 4. Predict
`python3 src/predict_entities.py`

# Best results (dev set, 10% of all data)
```json
{
        "learning_rate": 5e-5,
        "architecture": "BERT",
        "dataset": "NEREL-v1",
        "epochs": 5,
}
