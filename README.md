# wxbtool

A toolkit for WeatherBench based on PyTorch

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
