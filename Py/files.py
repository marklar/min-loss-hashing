"""
Filenames used by MLH code.
"""
data_dir = 'Data/'

# For sketching.
MEAN_VALS_FN   = data_dir + 'mean-vals.npy'
FEATURE_IDS_FN = data_dir + 'feature-ids.json'

# For evaluation.  This is just the root.  Add suffix for 'q'.
def hyperplanes_fn(n_bits):
    return '{0}W.q{1}.npy'.format(data_dir, n_bits)
