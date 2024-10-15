from pmwd.nbody import *


def observe(a_prev, a_next, ptcl, obsvbl, cosmo, conf):
    if obsvbl is None:
        pass
    else:
        mask = jnp.broadcast_to((obsvbl.mesh_a - a_prev) * (obsvbl.mesh_a - a_next) <=0, (conf.mesh_size, 3))
        #disp = obsvbl.disp.at[mask].set(
        #    ptcl.disp[mask] + (
        #        ptcl.vel[mask] * jnp.sum(ptcl.disp[mask] * obsvbl.mesh_los[mask], axis=1)
        #      / (obsvbl.mesh_drc[mask] * (obsvbl.mesh_a[mask] * obsvbl.mesh_E[mask]) - jnp.sum(ptcl.vel[mask] * obsvbl.mesh_los[mask], axis=1))
        #    ) 
        #)
        disp_proj = jnp.sum(ptcl.disp*obsvbl.mesh_los, axis=1, keepdims=True)
        vel_proj  = jnp.sum(ptcl.vel*obsvbl.mesh_los, axis=1, keepdims=True)
        tmp_qty = (obsvbl.mesh_drc * (obsvbl.mesh_a * obsvbl.mesh_E) - vel_proj)

        disp = jnp.where(mask, ptcl.disp + ptcl.vel * disp_proj / tmp_qty, obsvbl.disp)
        vel  = jnp.where(mask, ptcl.vel, obsvbl.vel)
        obsvbl = obsvbl.replace(disp=disp, vel=vel)
    return obsvbl


def observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf):
    # WIP; used to account for the obs gradients
    mask = jnp.broadcast_to((obsvbl.mesh_a - a_prev) * (obsvbl.mesh_a - a_next) <=0, (conf.mesh_size, 3))
    
    disp_proj = jnp.sum(ptcl.disp*obsvbl.mesh_los, axis=1, keepdims=True)
    vel_proj  = jnp.sum(ptcl.vel*obsvbl.mesh_los, axis=1, keepdims=True)
    tmp_qty = (obsvbl.mesh_drc * (obsvbl.mesh_a * obsvbl.mesh_E) - vel_proj)
    
    disp_obs_disp_snap = 1. + (ptcl.vel * obsvbl.mesh_los / tmp_qty)
    disp_obs_vel_snap  = disp_proj / tmp_qty - vel_proj * disp_proj / tmp_qty / tmp_qty

    disp_cot = ptcl_cot.disp + jnp.where(mask, obsvbl_cot.disp * disp_obs_disp_snap, 0)
    vel_cot  = ptcl_cot.vel  + jnp.where(mask, obsvbl_cot.disp * disp_obs_vel_snap 
                                             + obsvbl_cot.vel  , 0)
    
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot)
    return ptcl_cot
    

def observe_init(a, ptcl, obsvbl, cosmo, conf, obs_offset=None):
    if obsvbl is None or obs_offset is None:
        obsvbl = None
    else:
        obsvbl = obsvbl.set_obs(obs_offset=obs_offset, cosmo=cosmo)
    return obsvbl


@jit
def nbody_lightcone_init(a, ptcl, obsvbl, cosmo, conf, obs_offset):
    ptcl = force(a, ptcl, cosmo, conf)

    ptcl = coevolve_init(a, ptcl, cosmo, conf)

    obsvbl = observe_init(a, ptcl, obsvbl, cosmo, conf, obs_offset)

    return ptcl, obsvbl



@jit
def nbody_lightcone_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf):
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf)

    ptcl = coevolve(a_prev, a_next, ptcl, cosmo, conf)

    obsvbl = observe(a_prev, a_next, ptcl, obsvbl, cosmo, conf)

    return ptcl, obsvbl

@jit
def nbody_adj_lightcone_step(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    # order reversed
    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = integrate_adj(
        a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, conf)

    ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf)


    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force



@partial(custom_vjp, nondiff_argnums=(4, 5)) # we probably still need the offset as it is cosmology dependent
def nbody_lightcone(ptcl, obsvbl, cosmo, conf, reverse=False, obs_offset=None):
    """N-body time integration. Use jax.lax.scan to speed up the compilation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody_arr = jnp.array([a_nbody[:-1], a_nbody[1:]]).T

    ptcl, obsvbl = nbody_lightcone_init(a_nbody[0], ptcl, obsvbl, cosmo, conf, obs_offset)

    def _nbody_lightcone_step(carry, x):
        ptcl, obsvbl = carry
        a_prev, a_next = x
        ptcl, obsvbl = nbody_lightcone_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
        return (ptcl, obsvbl), None
    (ptcl, obsvbl), _ = scan(_nbody_lightcone_step, (ptcl, obsvbl), a_nbody_arr)
    return ptcl, obsvbl



@jit
def nbody_adj_lightcone_init(a, ptcl, ptcl_cot, obsvbl_cot, cosmo, conf):
    #ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo)

    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a, ptcl, ptcl_cot, cosmo, conf)

    cosmo_cot = tree_map(jnp.zeros_like, cosmo)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force

def nbody_adj_lightcone(ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf, reverse=False): #WIP
    """N-body time integration with adjoint equation. Use jax.lax.scan to speed up the compilation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody_arr = jnp.array([a_nbody[:0:-1], a_nbody[-2::-1]]).T


    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        a_nbody[-1], ptcl, ptcl_cot, obsvbl_cot, cosmo, conf)

    def _nbody_adj_step(carry, x):
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = carry
        a_prev, a_next = x
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_lightcone_step(
            a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
        return (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), None
    
    (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), _ = scan(_nbody_adj_step, (ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force), a_nbody_arr)

    return ptcl, ptcl_cot, cosmo_cot

def nbody_fwd_lightcone(ptcl, obsvbl, cosmo, conf, reverse, obs_offset):
    ptcl, obsvbl = nbody_lightcone(ptcl, obsvbl, cosmo, conf, reverse, obs_offset)
    return (ptcl, obsvbl), (ptcl, obsvbl, cosmo, conf)


def nbody_bwd_lightcone(reverse, obs_offset, res, cotangents):
    # `reverse` is a nodiff argument.
    ptcl, obsvbl, cosmo, conf = res
    ptcl_cot, obsvbl_cot = cotangents
    nbody_adj_lightcone 

    ptcl, ptcl_cot, cosmo_cot = nbody_adj_lightcone(ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf,
                                          reverse=reverse)

    return ptcl_cot, obsvbl_cot, cosmo_cot, None

nbody_lightcone.defvjp(nbody_fwd_lightcone, nbody_bwd_lightcone)
