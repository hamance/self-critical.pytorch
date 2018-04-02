from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import tqdm
import numpy as np

def main(params):
    inp_dir = params['input_dir']
    # fc_dir = os.path.join(inp_dir, 'cocotalk_fc')
    # att_dir = os.path.join(inp_dir, 'cocotalk_att')
    # if not os.path.exists(fc_dir):
    #     raise ValueError("feat dir not found : %s" % fc_dir)
    # if not os.path.exists(att_dir):
    #     raise ValueError("feat dir not found : %s" % att_dir)

    feat_type = params['type']
    fpaths = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir)]
    fp2id = {}

    feat_mmp = None

    for idx, fp in tqdm.tqdm(enumerate(fpaths), total=len(fpaths)):
        if feat_type == 'fc':
            feat = np.load(fp)
        elif feat_type == 'att':
            feat = np.load(fp)['feat']
        else:
            raise ValueError("Invalid feat type, should be fc or att.")

        if feat_mmp is None:
            mmp_shape = (len(fpaths),) + feat.shape
            print("mmp shape: ", mmp_shape)
            feat_mmp = np.memmap(params['output_path'] + '.mmp', dtype='float32', mode='w+', shape=mmp_shape)

        feat_mmp[idx] = feat
        fp2id[fp] = idx

    with open(params['output_path'] + '.json', 'wb') as f:
        json.dump(fp2id, f)

    print("Done.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # feature_dir
    parser.add_argument('--type', required=True, help='input feature directory')
    parser.add_argument('--input_dir', required=True, help='input feature directory')
    parser.add_argument('--output_path', required=True, help='output feature memmap file')
    # parser.add_argument('--feat_shape', required=True, help='intput feat shape')

    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
