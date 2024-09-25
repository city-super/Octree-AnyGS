python train.py --config config/scaffoldgs/base_model.yaml --port 4001 &
sleep 20s

python train.py --config config/scaffoldgs/lod_model.yaml --port 4002 &
sleep 20s

python train.py --config config/3dgs/base_model.yaml --port 4003 &
sleep 20s

python train.py --config config/3dgs/lod_model.yaml --port 4004 &
sleep 20s

python train.py --config config/2dgs/base_model.yaml --port 4005 &
sleep 20s

python train.py --config config/2dgs/lod_model.yaml --port 4006 &