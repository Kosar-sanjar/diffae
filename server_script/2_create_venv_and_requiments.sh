CURRENT_PATH=$(pwd)
cd ..

python3.8 -m venv venv
source venv/bin/activate
#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements_server.txt
#pip install Pillow==9.5.0
cat requirements_server.txt | xargs -t -L 1 -P3 pip download -q
cat requirements_server.txt | xargs -t -L 1 -P1 pip install -q

cd $CURRENT_PATH
pip install gdown
