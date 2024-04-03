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

* Run `python3 src/nerel_to_json.py` to build datasets
* In folder `./dataset` you can find `nerel_data_train.json`, `nerel_data_dev.json` and `nerel_data_class_names.json`
* Dataset configuration can be find in `constants.py`

# Training

  
# Architecture


# Sentence-embeddings



# Export to onnx


# Best results (dev set, 10% of all data)
```json
{
    "dropout": 0.33217146571770845,
    "beta": 0.9858882365915637,
    "gamma": 0.46713439384680727,
    "learning_rate": 0.00012052080975210854,
    "base_model": "google/electra-small-discriminator",
    "model_type": "electra",
    "batch_size": 128
}
