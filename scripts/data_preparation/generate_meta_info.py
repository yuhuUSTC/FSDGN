from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = r'D:\VideoDehazing\Data\REVIDE_indoor\Test\gt'
    meta_info_txt = r'D:\VideoDehazing\BasicSR\basicsr\data\meta_info\REVIDE_indoor_test_GT.txt'

    img_list = sorted(list(scandir(gt_folder, suffix='JPG', recursive=True)))

    keys = [v.split('\\')[0] for v in img_list]  # example: 000/00000000
    key = sorted(set(keys))
    with open(meta_info_txt, 'w') as f:
        for item in key:
            info = f'{item} {keys.count(item)} ({1800},{1806},{3})'
            f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_div2k()
