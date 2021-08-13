# wxbtool

[![DOI](https://zenodo.org/badge/269931312.svg)](https://zenodo.org/badge/latestdoi/269931312)

A toolkit for WeatherBench based on PyTorch

Install
--------

```bash
pip install wxbtool
```

Cheat sheet
-----------

Start a data set server for 3-days prediction of t850 by Weyn's solution
```bash
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d
```

Start a training process for a UNet model following Weyn's solution
```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

Start a testing process for a UNet model following Weyn's solution
```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
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
