# Installation of Tensorflow for MAC M1 chip
To install tensorflow on MAC M1, just run the following pip commands from the project root directory:
```shell
pip uninstall tensorflow
gdown 1gwU0tDUQcGaFWf9lrmWE1y7BEaRqsEtu -O tf-mac-m1/
pip install tf-mac-m1/tensorflow-1.15.0-py3-none-any.whl
```
It will install Tensorflow 1.15 through a wheel file built from Rosetta 2. This file was provided by **Isa Milefchik** via this Google Drive <a href="https://drive.google.com/drive/folders/11cACNiynhi45br1aW3ub5oQ6SrDEQx4p?usp=sharing">link</a>.