echo "Training resnet"
python3 train_resnet.py
echo "Finished training resnet"

echo "---------------"

echo "Traiing stylenet"

python3 stylenet.py

echo "Finished training stylenet"
