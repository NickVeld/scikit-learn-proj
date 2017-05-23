# Реализация алгоритма спектральных вложений Грассмана-Штифеля
## в рамках библиотеки scikit-learn
### Вельдяйкин Николай Олегович, Группа 154
The project over 2 year of HSE Faculty of Computer Science 2016-2017

Задача актуальна, так как она дополняет библиотеку достаточно [извеcтными](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.5053&rep=rep1&type=pdf) инструментом, и в случае публикации даст новые возможности в manifold learning.

Так как нам дана бибилиотека, которая доллжна исползоваться, то инструменты такие, которые подходят для работы с этой бибилиотекой: интерпретатор Python с нужными зависимостями, сама библиотека scikit-learn, как основа и GitHub. 

#### Контрольные точки:

##### План:

КТ1: Документация по проекту и репозиторий

КТ2: "Out of sample" для алгоритма Spectral Embedding: для "nearest_neighbors" метода поиска affinity matrix

КТ3: "Inverse out of sample" для реализованого алгоритма к КТ2 

##### Выполнено:
- Выложена документация к проекту
- Разобрана теоритическая основа проекта
- Реализован метод получения вложения для новых точек ("Out of sample")
- Выстроена логика тестирования моего вклада в эту бибилиотеку
- Протестирована реализация "Out of sample"
- Реализован метод получения точки в исходной размерности пространства для новых точек вложения  ("Inverse out of sample")
- Протестирована реализация "Inverse out of sample"

#### Архитектура:
Исходная точка репозитория - дубликат репозитория scikit-learn.

Были изменены следующие объекты:
- [spectral_embedding_.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/sklearn/manifold/spectral_embedding_.py)
- [mds.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/sklearn/manifold/mds.py)

Были добавлены следующие объекты:
- [директория mytests](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests)
 [скрипт-установщик/переустановщик reinstall.sh](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/reinstall.sh)
    * [генератор выборки dataset_generator.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/dataset_generator.py)
    * [файл с тестирующим SpectralEmbedding кодом se_test.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/se_test.py)
- README.md (этот файл)

#### Инструкция:
##### Подготовка
###### Способ 1, как задумывалось, но этот способ не для всех:
*Этот способ у меня не заработал на трех из трех операционных системах семейства Ubuntu*

Так как я дописываю библиотеку scikit-learn, то наилучшая документация находится в репозитории scikit-learn и на страницах бибилиотеки. У меня хранится копия [инструкции по работе с исходным кодом](https://github.com/NickVeld/scikit-learn-proj/blob/master/README.rst) (раздел development).

###### Способ 2, стабильный, но нестандартный:
Есть следующая [инструкция по установке](http://scikit-learn.org/stable/developers/advanced_installation.html#linux):
- Устанавливаем необходимые для сборки компоненты 
- Переходим к [параграфу для сборки из исходного кода](http://scikit-learn.org/stable/developers/advanced_installation.html#from-source-package)
- Скачиваем исходный код по предложенной ссылке на [pypi](https://pypi.python.org/pypi/scikit-learn)
- В распакованном коде подменяем файлы с кодом файлами, которые модифицировались, из этого репозитория, на данный момент это: [spectral_embedding_.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/sklearn/manifold/spectral_embedding_.py), ~~[mds.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/sklearn/manifold/mds.py) (Для MDS сделаны только сигнатуры,но не алгоритм)~~
- Изучите скрипт [reinstall.sh в mytests](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/reinstall.sh) и при необходимости адаптируте под свою систему. **Неизвестно, что будет, если к моменту запуска этого скрипта или сборке по [инструкции](http://scikit-learn.org/stable/developers/advanced_installation.html#from-source-package) есть установленный иным способом sklearn!**
- Первый раз можно установить, как строкой "python3 setup.py install", так и запуском скрипта из корневой директории скачанного с pypi sklearn *(скрипт запускать с правами администратора)*. **Работа на python2 не гарантируется!**
- При необходимости переустановки запустите скрипт из корневой директории скачанного с pypi sklearn с правами администратора

##### Эксплуатация
После выполнения шагов по подготовке в собственном python-коде можно написать "import sklearn", "import sklearn.manifold" или "from sklearn.manifold import SpectralEmbedding" *(если нужно использовать написанное мной)*, чтобы использовать scikit-learn

##### Тестирование
**Работа на python2 и всех системах кроме Ubuntu 14.04.5 Desktop и старше не гарантируется!**
Для тестирования есть два модуля в папке mytests ([генератор выборки dataset_generator.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/dataset_generator.py) и [файл с тестирующим SpectralEmbedding кодом se_test.py](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/se_test.py), они могут быть в любом месте, но главное, чтобы в одной директории.

В директории с этими модулями написать "python3 se_test.py *аргументы*", где первый опциональный аргумент может принимать значение "list" (показывается результат работы всех генераторов) или "show" *(показать результат работы n-ого генератора, n - целочисленное неотрицательное значение второго аргумента)*. "python3 se_test.py" эквивалентно "0". Если первый аргумент "show" и не указан второй аргумент или указан неправильно, то поведение неопределенно. В остальных случаях программа завершится, ничего не сделав.

Генераторы:
0. Базовый, согнутый участок плосоксти
1. Труба ширины 1 и радиусом 2
2. Сдвинутое кольцо ширины 1 
3. Спираль ширины 1 из одного витка
4. Сдвинутая спираль ширины 1 из одного витка
5. Сдвинутая спираль ширины 2000 из одного витка
6. Лента Мёбиуса ширины 1 и радиуса 1
7. Плоскость, согнутая в форме S
8. Плоскость, согнутая в спираль

Пример: "python3 se_test.py show 7"
Сверху слева исходная выборка, сверху второе изображение - это исходная выборка + 1000 новых точки, сверху третье изображение -  исходная выборка, снизу слева вложение, полученное имеющимся алгоритмом, снизу в центре вложение новых точек, наложенных на вложение исходных точек, вложение с помощью OoS исходной выборки, наложенное на вложение исходной выборки (для оценки работы алгоритма, получилось хорошо).
![S_curve](https://github.com/NickVeld/scikit-learn-proj/blob/master/mytests/images/S_1000_1000_1000_inverse.PNG)
