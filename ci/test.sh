set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt --user; fi
python3 setup.py install
python3 -m pytest examples/pytorch2msnhnet
