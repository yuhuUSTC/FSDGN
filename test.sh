optpath='FSDGN/basicsr/options/test/dehaze_test.yml'
export PYTHONPATH=/bfs/yuhu/code/FSDGN:$PYTHONPATH
CUDA_VISIBLE_DEVICES="0" python /bfs/yuhu/code/FSDGN/basicsr/test.py -opt $optpath
