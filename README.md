# wxbtool

A toolkit for WeatherBench based on PyTorch

Warning: This project is at its early stage and the api is not very stable

Install
--------

```bash
pip install wxbtool
```

How to use
-----------

* [quick start](https://github.com/caiyunapp/wxbtool/wiki/quick-start)
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
