CURRENT_PATH=$(pwd)
cd ..

source venv/bin/activate
pip install gdown

mkdir datasets/ffhq256.lmdb
gdown "https://drive.google.com/uc?id=1ZPs9e5_RxZ-JL36udRzWsULv0mpo_Hhv" -O eeg_5_95_std.pth
mv eeg_5_95_std.pth datasets

cd $CURRENT_PATH
