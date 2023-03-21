import nlopt

def optimiser(k, fun, xtol, lb, ub):
    '''
    This is a wrapper function for the NLOpt optimiser class. 
    Inputs:
    k: Initialisation array
    fun: function for optimising
    xtol: relative optimiser tolerance
    lb: lower bounds array
    ub: upper bounds array
    '''

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(k))
    opt.set_min_objective(fun)
    opt.set_xtol_rel(xtol)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    x = opt.optimize(k)

    return  x, opt.last_optimum_value()
