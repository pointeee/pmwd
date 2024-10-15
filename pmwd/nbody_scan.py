from pmwd.nbody import *

@partial(custom_vjp, nondiff_argnums=(4,))
def nbody_scan(ptcl, obsvbl, cosmo, conf, reverse=False):
    """N-body time integration. Use jax.lax.scan to speed up the compilation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody_arr = jnp.array([a_nbody[:-1], a_nbody[1:]]).T

    ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)

    def _nbody_step(carry, x):
        ptcl, obsvbl = carry
        a_prev, a_next = x
        ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
        return (ptcl, obsvbl), None
    (ptcl, obsvbl), _ = scan(_nbody_step, (ptcl, obsvbl), a_nbody_arr)
    return ptcl, obsvbl

def nbody_adj_scan(ptcl, ptcl_cot, obsvbl_cot, cosmo, conf, reverse=False):
    """N-body time integration with adjoint equation. Use jax.lax.scan to speed up the compilation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody_arr = jnp.array([a_nbody[:0:-1], a_nbody[-2::-1]]).T


    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        a_nbody[-1], ptcl, ptcl_cot, obsvbl_cot, cosmo, conf)

    def _nbody_adj_step(carry, x):
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = carry
        a_prev, a_next = x
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_step(
            a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
        return (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), None
    
    (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), _ = scan(_nbody_adj_step, (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), a_nbody_arr)

    return ptcl, ptcl_cot, cosmo_cot


def nbody_fwd_scan(ptcl, obsvbl, cosmo, conf, reverse):
    ptcl, obsvbl = nbody_scan(ptcl, obsvbl, cosmo, conf, reverse)
    return (ptcl, obsvbl), (ptcl, cosmo, conf)
    
def nbody_bwd_scan(reverse, res, cotangents):
    ptcl, cosmo, conf = res
    ptcl_cot, obsvbl_cot = cotangents

    ptcl, ptcl_cot, cosmo_cot = nbody_adj_scan(ptcl, ptcl_cot, obsvbl_cot, cosmo, conf,
                                          reverse=reverse)

    return ptcl_cot, obsvbl_cot, cosmo_cot, None


nbody_scan.defvjp(nbody_fwd_scan, nbody_bwd_scan)
