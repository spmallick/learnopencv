BATCH_SIZE=64
NUM_WORKERS=8
SIZE=[480,640]
IMGPATH_FILE='./imagenetsplitpaths.txt'
SOFT_TARGET_PATH='./resnet152_results.npy'
TEMPERATURE=3
EPOCHS=10
SAVE_PATH='./resnet_{}.pt'
EVAL_INTERVAL=1000
LR=1e-4

LOSS_NET_PATH='./models/resnet_9.pt'
STYLE_TARGET='./manulogo.png'
