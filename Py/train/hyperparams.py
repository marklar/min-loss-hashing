
class Hyperparams:
    """ For setting parameters rho and shrink_w.
        Rho is in the hinge loss fn.
        Shrink_w is the weight decay, during training.
    """
    n_iters = 75
    lambduh = 0.5
    n_validations_during = 0
    n_validations_after = 5

    def __init__(data, n_bits):
        self.data   = data
        self.n_bits = n_bits
        self.rho      = self.calc_rho()
        self.shrink_w = self.calc_shrink_w()

    def calc_rho(self):
        return self.update_rho(self.estimate_rho())

    def calc_shrink_w(self, rho):
        """ Use validation to update shrink_w (weight decay). """
        opts = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        pairs = [(self.ave_precision(rho, shrink_w), shrink_w) for shrink_w in opts]
        return max(pairs)[1]

    # --- RHO helpers ---

    def estimate_rho(self):
        """ Using simple LSH, find recall.  Use that to estimate rho."""
        # Create 'train' data from 'real' data (??)
        val_data = self.create_training_data()
        
        # Use LSH, and find recall for each Hamming dist.
        # recall_vals: list.  at each index (Hamming dist), a recall val.
        #
        # --- FIXME ---
        (_, recall_vals) = eval_lsh(self.n_bits, val_data)
        
        # Why with LOW recall?
        return [r < 0.3 for r in recall_vals].count(True)

    def update_rho(self, rho):
        """ Rho might get large.  Set manually if want smaller. """
        # Run MLH for a number of different rho options.
        pairs = [(self.ave_precision(r, 1e-3), r) for r in self.rho_opts(rho)]
        # Find the rho which has the best MAP (mean ave precision).
        return max(pairs)[1]
        # return best_rho or estimated_rho

    def rho_opts(self, rho):
        """ Create as many as 5 positive options. """
        step = max(1, self.n_bits / 32)
        opts = [rho + (i * step) for i in range(-2, 3)]
        return filter(lambda i: i >= 1, opts)

    ## --- util ---

    # --- FIXME ---
    def create_training_data(self):
        # self.data, 'train', True
        pass

    # --- FIXME ---
    def ave_precision(rho, shrink_w):
        return mlh(self.data,
                   rho,
                   self.lambduh,
                   self.n_bits,
                   'train',
                   self.n_iters,
                   self.n_validations_during,
                   self.n_validations_after,
                   shrink_w,
                   False      # i.e. don't shrink learning rate
                   ).ave_precision
