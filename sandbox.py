from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.run_handler import RunHandler
from src.logger import DataLogger
from src import util
from src import visuals
from src.sparse import Sparsity
import numpy as np
import time
import matplotlib.pyplot as plt

orig = util.load_hsi("raw\\Indian_pines_corrected.mat")

minval, maxval, depth = util.get_hsi_statistics(orig)
norm_orig = util.normalize_zero_mean(orig, minval, maxval)

dct = util.DCTBasis()
axes = [0, 1, 2]

trans = norm_orig.copy()
for ax in axes:
    trans = dct.forward(trans, axis=ax)

max_trans = np.max(np.abs(trans))
norm_trans = (trans + max_trans) / (2 * max_trans)      # normalization to [0, 1]

max_int = (1 << depth) - 1
quant = np.clip(np.round(norm_trans * max_int).astype(np.uint64), 0, max_int)

norm_trans_rec = (quant.astype(np.float64) / max_int)
trans_rec = norm_trans_rec * 2 * max_trans - max_trans

norm_rec = trans_rec.copy()
for ax in axes:
    norm_rec = dct.inverse(norm_rec, axis=ax)

rec = util.denormalize_zero_mean(norm_rec, minval, maxval)

print(f'RMSE: {util.calc_rmse(rec, orig)}')
print(f'PSNR: {util.calc_psnr(rec, orig, depth)}')
print(f'SAM: {util.calc_sam(rec, orig)}')

bands = [80]
plt.figure()
for i, b in enumerate(bands):
    plt.subplot(i+1, 3, 1 + 3*i)
    plt.imshow(orig[:,:,b], cmap='gray')
    plt.subplot(i+1, 3, 2 + 3*i)
    plt.imshow(trans[:,:,b], cmap='gray')
    plt.subplot(i+1, 3, 3 + 3*i)
    plt.imshow(rec[:,:,b], cmap='gray')

plt.tight_layout()
plt.show()

