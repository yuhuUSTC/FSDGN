optpath='FSDGN/basicsr/options/train/dehaze.yml'
export PYTHONPATH=/bfs/yuhu/code/FSDGN:$PYTHONPATH
CUDA_VISIBLE_DEVICES="0" python /bfs/yuhu/code/FSDGN/basicsr/train.py -opt $optpath
