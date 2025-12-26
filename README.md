# Meme Detector

===============

## Что это?
Это система распознавания популярных мемов по позе тела и мимике в реальном времени. Когда вы изображаете мем, программа распознаёт его и показывает соответствующее изображение.

----------

## Структура пакета

`meme_detector.py` — основной скрипт для детекции мемов в реальном времени.

`notebooks/` — Jupyter‑ноутбуки для обучения и экспериментов.

`src/` — модули для предобработки данных, моделей и вспомогательных функций.

`models/` — сохранённые веса моделей.

`data/` — обучающие видеоданные.

`resources/memes/` — изображения мемов.

----------

## Установка и использование
1. Установите необходимые библиотеки:
   ```bash
   pip install -r requirements.txt
   ```
2. Запустите детектор.
   ```bash
   python meme_detector.py
   ```

## Как это работает
Модель использует MediaPipe Pose для извлечения ключевых точек тела и FaceMesh для извлечения ключевых точек лица. Извлеченные признаки классифицируются с использованием SVM (Support Vector Machine) с ядром RBF.

----------

License
-------

WTFNMFPLv1

```
    DO WHAT THE FUCK YOU WANT TO BUT IT'S NOT MY FAULT PUBLIC LICENSE
                    Version 1, October 2013

 Copyright © 2013 Ben McGinnes <ben@adversary.org>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

   DO WHAT THE FUCK YOU WANT TO BUT IT'S NOT MY FAULT PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.

  1. Do not hold the author(s), creator(s), developer(s) or
     distributor(s) liable for anything that happens or goes wrong
     with your use of the work.
```