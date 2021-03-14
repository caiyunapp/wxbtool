# wxbtool

A toolkit for WeatherBench based on PyTorch (Work in progress)

Install
--------

```bash
pip install wxbtool
```

How to use
-----------

* quick start
* understanding the physical process by plotting
* develop your own neural model
* try a toy physical model
* explore the possibility to combine neural and physical model together

How to release
---------------

```bash
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*

git tag va.b.c master
git push origin va.b.c
```

Contributors
------------

* Mingli Yuan ([Mountain](https://github.com/mountain))
* Ren Lu
