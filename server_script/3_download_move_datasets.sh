CURRENT_PATH=$(pwd)
cd ..

source venv/bin/activate
pip install gdown

mkdir datasets/ffhq256.lmdb

gdown "https://drive.google.com/uc?id=1ZPs9e5_RxZ-JL36udRzWsULv0mpo_Hhv" -O eeg_5_95_std.pth
mv eeg_5_95_std.pth datasets

gdown "https://drive.google.com/uc\?id\=1ZQaX_-jtW8nwRd6PiPv_UN0nI9GhgwEq" -O block_splits_by_image_all.pth
mv block_splits_by_image_all.pth datasets

gdown https://drive.google.com/uc\?id\=1fc_wDKQul6PakR2Dtg4sKeUhVaoLM6xz -O images.zip 
mv images.zip datasets
unzip datasets/images.zip -d datasets/images 

cd $CURRENT_PATH
