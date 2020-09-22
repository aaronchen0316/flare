import numpy as np
from numba import njit
from math import exp
import sys
import os
from flare.env import AtomicEnvironment
import flare.kernels.cutoffs as cf
from flare.kernels.kernels import (
    force_helper,
    grad_constants,
    grad_helper,
    force_energy_helper,
    three_body_en_helper,
    three_body_helper_1,
    three_body_helper_2,
    three_body_grad_helper_1,
    three_body_grad_helper_2,
    k_sq_exp_double_dev,
    k_sq_exp_dev,
    k2_sq_exp,
    k2_sq_exp_dev,
    k2_sq_exp_double_dev_1,
    k2_sq_exp_double_dev_2,
    k2_sq_exp_grad_ls,
    k2_sq_exp_dev_grad_ls,
    k2_sq_exp_double_dev_1_grad_ls,
    k2_sq_exp_double_dev_2_grad_ls,
    coordination_number,
    q_value,
    q_value_mc,
    mb_grad_helper_ls_,
    mb_grad_helper_ls_,
    three_body_se_perm,
    three_body_sf_perm,
    three_body_ss_perm,
    q_value_mc,
    mb_grad_helper_ls_,
    mb_grad_helper_ls,
)
from flare.kernels import two_body_mc_simple, three_body_mc_simple
from typing import Callable


def many_body_mc(
    env1: AtomicEnvironment,
    env2: AtomicEnvironment,
    d1,
    d2,
    hyps: "ndarray",
    cutoffs: "ndarray",
    cutoff_func: Callable = cf.quadratic_cutoff,
) -> float:

    return many_body_mc_jit(
        env1.q_array,
        env2.q_array,
        env1.q_neigh_array,
        env2.q_neigh_array,
        env1.bond_array_mb,
        env2.bond_array_mb,
        env1.q_neigh_grads,
        env2.q_neigh_grads,
        env1.neigh_array,
        env2.neigh_array,
        env1.q_neigh2_array,
        env2.q_neigh2_array,
        env1.ctype,
        env2.ctype,
        env1.etypes_mb,
        env2.etypes_mb,
        env1.neigh_etypes,
        env2.neigh_etypes,
        env1.unique_species,
        env2.unique_species,
        hyps[0],
        hyps[1],
        d1,
        d2,
        cutoffs[2],
        cutoff_func,
    )


def many_body_mc_grad(
    env1: AtomicEnvironment,
    env2: AtomicEnvironment,
    d1,
    d2,
    hyps: "ndarray",
    cutoffs: "ndarray",
    cutoff_func: Callable = cf.quadratic_cutoff,
) -> float:

    return many_body_mc_grad_jit(
        env1.q_array,
        env2.q_array,
        env1.q_neigh_array,
        env2.q_neigh_array,
        env1.bond_array_mb,
        env2.bond_array_mb,
        env1.q_neigh_grads,
        env2.q_neigh_grads,
        env1.neigh_array,
        env2.neigh_array,
        env1.q_neigh2_array,
        env2.q_neigh2_array,
        env1.ctype,
        env2.ctype,
        env1.etypes_mb,
        env2.etypes_mb,
        env1.neigh_etypes,
        env2.neigh_etypes,
        env1.unique_species,
        env2.unique_species,
        hyps[0],
        hyps[1],
        d1,
        d2,
        cutoffs[2],
        cutoff_func,
    )


def many_body_mc_force_en(
    env1: AtomicEnvironment,
    env2: AtomicEnvironment,
    d1,
    hyps: "ndarray",
    cutoffs: "ndarray",
    cutoff_func: Callable = cf.quadratic_cutoff,
) -> float:

    return many_body_mc_force_en_jit(
        env1.q_array,
        env2.q_array,
        env1.q_neigh_array,
        env2.q_neigh_array,
        env1.bond_array_mb,
        env2.bond_array_mb,
        env1.q_neigh_grads,
        env1.neigh_array,
        env1.q_neigh2_array,
        env1.ctype,
        env2.ctype,
        env1.etypes_mb,
        env2.etypes_mb,
        env1.neigh_etypes,
        env1.unique_species,
        env2.unique_species,
        hyps[0],
        hyps[1],
        d1,
        cutoffs[2],
        cutoff_func,
    )


def many_body_mc_en(
    env1: AtomicEnvironment,
    env2: AtomicEnvironment,
    hyps: "ndarray",
    cutoffs: "ndarray",
    cutoff_func: Callable = cf.quadratic_cutoff,
) -> float:

    return many_body_mc_en_jit(
        env1.q_array,
        env2.q_array,
        env1.q_neigh_array,
        env2.q_neigh_array,
        env1.bond_array_mb,
        env2.bond_array_mb,
        env1.ctype,
        env2.ctype,
        env1.etypes_mb,
        env2.etypes_mb,
        env1.unique_species,
        env2.unique_species,
        hyps[0],
        hyps[1],
        cutoffs[2],
        cutoff_func,
    )


@njit
def many_body_mc_grad_jit(
    q_array_1,
    q_array_2,
    q_neigh_array_1,
    q_neigh_array_2,
    bond_array_1,
    bond_array_2,
    q_neigh_grads_1,
    q_neigh_grads_2,
    neigh_bond_array_1,
    neigh_bond_array_2,
    q_neigh2_array_1,
    q_neigh2_array_2,
    c1,
    c2,
    etypes1,
    etypes2,
    neigh_etypes1,
    neigh_etypes2,
    species1,
    species2,
    sig,
    ls,
    d1,
    d2,
    r_cut,
    cutoff_func,
):
    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8
    )
    kern = 0.0
    ls_derv = 0.0

    for s in useful_species:
        s1 = np.where(species1 == s)[0][0]
        s2 = np.where(species2 == s)[0][0]

        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        # this has been calculated in env_getarray.py
        dq1_dr = 0.0
        for m in range(q_neigh_array_1.shape[0]):
            e1 = etypes1[m]
            if e1 == s:
                dq1_dr += q_neigh_grads_1[m, d1 - 1]

        dq2_dr = 0.0
        for n in range(q_neigh_array_2.shape[0]):
            e2 = etypes2[n]
            if e2 == s:
                dq2_dr += q_neigh_grads_2[n, d2 - 1]

        for m in range(q_neigh_array_1.shape[0]):
            qn1 = q_neigh_array_1[m, s1]
            ri = bond_array_1[m, 0]
            ci = bond_array_1[m, d1]
            fi, dfi = cutoff_func(r_cut, ri, ci)
            e1 = etypes1[m]

            for n in range(q_neigh_array_2.shape[0]):
                qn2 = q_neigh_array_2[n, s2]
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, dfj = cutoff_func(r_cut, rj, cj)
                e2 = etypes2[n]

                if c1 == c2 and e1 == e2:
                    kq = k2_sq_exp(q1, q2, qn1, qn2, sig, ls)
                    kern += kq * dfj * dfi  # 0-0
                    ls_term = k2_sq_exp_grad_ls(q1, q2, qn1, qn2, sig, ls)
                    ls_derv += ls_term * dfj * dfi

                    dk_dq1 = k2_sq_exp_dev(q1, q2, qn1, qn2, sig, ls)
                    kern += dq1_dr * dk_dq1 * fi * dfj  # 1-0
                    ls_term = k2_sq_exp_dev_grad_ls(q1, q2, qn1, qn2, sig, ls)
                    ls_derv += ls_term * dq1_dr * fi * dfj

                    dk_dq2 = k2_sq_exp_dev(q2, q1, qn2, qn1, sig, ls)
                    kern += dq2_dr * dk_dq2 * dfi * fj  # 0-1
                    ls_term = k2_sq_exp_dev_grad_ls(q2, q1, qn2, qn1, sig, ls)
                    ls_derv += dq2_dr * ls_term * dfi * fj  # 0-1

                    dk_dq1q2 = k2_sq_exp_double_dev_1(q1, q2, qn1, qn2, sig, ls)  # 1-1
                    kern += dk_dq1q2 * dq1_dr * dq2_dr * fi * fj
                    ls_term = k2_sq_exp_double_dev_1_grad_ls(
                        q1, q2, qn1, qn2, sig, ls
                    )  # 1-1
                    ls_derv += ls_term * dq1_dr * dq2_dr * fi * fj

                if c1 == e2 and c2 == e1:
                    kq = k2_sq_exp(q1, qn2, qn1, q2, sig, ls)
                    kern += kq * dfj * dfi
                    ls_term = k2_sq_exp_grad_ls(q1, qn2, qn1, q2, sig, ls)
                    ls_derv += ls_term * dfj * dfi

                    dk_dq1 = k2_sq_exp_dev(q1, qn2, qn1, q2, sig, ls)
                    kern += dq1_dr * dk_dq1 * fi * dfj
                    ls_term = k2_sq_exp_dev_grad_ls(q1, qn2, qn1, q2, sig, ls)
                    ls_derv += dq1_dr * ls_term * fi * dfj

                    dk_dq2 = k2_sq_exp_dev(q2, qn1, q1, qn2, sig, ls)
                    kern += dq2_dr * dk_dq2 * dfi * fj
                    ls_term = k2_sq_exp_dev_grad_ls(q2, qn1, q1, qn2, sig, ls)
                    ls_derv += dq2_dr * ls_term * dfi * fj

                    dk_dq1q2 = k2_sq_exp_double_dev_2(q1, qn2, q2, qn1, sig, ls)
                    kern += dk_dq1q2 * dq1_dr * dq2_dr * fi * fj
                    ls_term = k2_sq_exp_double_dev_2_grad_ls(q1, qn2, q2, qn1, sig, ls)
                    ls_derv += ls_term * dq1_dr * dq2_dr * fi * fj

                # 2nd neighbors
                if c1 == s:
                    dqn1_dr = q_neigh_grads_1[m, d1 - 1]
                    neigh2_array_1 = q_neigh2_array_1[m]
                    for i in range(len(neigh2_array_1)):
                        q2n1 = neigh2_array_1[i, s1]
                        en1 = neigh_etypes1[m][i]
                        rni = neigh_bond_array_1[m][i, 0]
                        fni, _ = cutoff_func(r_cut, rni, 0)

                        if e1 == c2 and en1 == e2:
                            dk_dq = k2_sq_exp_dev(qn1, q2, q2n1, qn2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * dfj  # 2-0
                            ls_term = k2_sq_exp_dev_grad_ls(qn1, q2, q2n1, qn2, sig, ls)
                            ls_derv += ls_term * dqn1_dr * fni * dfj  # 2-0

                            dk_dqn1q2 = k2_sq_exp_double_dev_1(
                                qn1, q2, q2n1, qn2, sig, ls
                            )
                            kern += dk_dqn1q2 * dqn1_dr * dq2_dr * fni * fj  # 2-1
                            ls_term = k2_sq_exp_double_dev_1_grad_ls(
                                qn1, q2, q2n1, qn2, sig, ls
                            )
                            ls_derv += ls_term * dqn1_dr * dq2_dr * fni * fj  # 2-1

                        if en1 == c2 and e1 == e2:
                            dk_dq = k2_sq_exp_dev(qn1, qn2, q2n1, q2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * dfj
                            ls_term = k2_sq_exp_dev_grad_ls(qn1, qn2, q2n1, q2, sig, ls)
                            ls_derv += ls_term * dqn1_dr * fni * dfj

                            dk_dqn1q2 = k2_sq_exp_double_dev_2(
                                qn1, qn2, q2, q2n1, sig, ls
                            )
                            kern += dk_dqn1q2 * dqn1_dr * dq2_dr * fni * fj
                            ls_term = k2_sq_exp_double_dev_2_grad_ls(
                                qn1, qn2, q2, q2n1, sig, ls
                            )
                            ls_derv += ls_term * dqn1_dr * dq2_dr * fni * fj

                # 2nd neighbors
                if c2 == s:
                    dqn2_dr = q_neigh_grads_2[n, d2 - 1]
                    neigh2_array_2 = q_neigh2_array_2[n]
                    for j in range(len(neigh2_array_2)):
                        q2n2 = neigh2_array_2[j, s2]
                        en2 = neigh_etypes2[n][j]
                        rnj = neigh_bond_array_2[n][j, 0]
                        fnj, _ = cutoff_func(r_cut, rnj, 0)

                        if e2 == c1 and en2 == e1:
                            dk_dq = k2_sq_exp_dev(qn2, q1, q2n2, qn1, sig, ls)
                            kern += dk_dq * dqn2_dr * dfi * fnj  # 0-2
                            ls_term = k2_sq_exp_dev_grad_ls(qn2, q1, q2n2, qn1, sig, ls)
                            ls_derv += ls_term * dqn2_dr * dfi * fnj  # 0-2

                            dk_dq1qn2 = k2_sq_exp_double_dev_1(
                                qn2, q1, q2n2, qn1, sig, ls
                            )
                            kern += dk_dq1qn2 * dq1_dr * dqn2_dr * fi * fnj  # 1-2
                            ls_term = k2_sq_exp_double_dev_1_grad_ls(
                                qn2, q1, q2n2, qn1, sig, ls
                            )
                            ls_derv += ls_term * dq1_dr * dqn2_dr * fi * fnj  # 1-2

                        if en2 == c1 and e2 == e1:
                            dk_dq = k2_sq_exp_dev(qn2, qn1, q2n2, q1, sig, ls)
                            kern += dk_dq * dqn2_dr * dfi * fnj
                            ls_term = k2_sq_exp_dev_grad_ls(qn2, qn1, q2n2, q1, sig, ls)
                            ls_derv += ls_term * dqn2_dr * dfi * fnj

                            dk_dq1qn2 = k2_sq_exp_double_dev_2(
                                qn2, qn1, q1, q2n2, sig, ls
                            )
                            kern += dk_dq1qn2 * dq1_dr * dqn2_dr * fi * fnj
                            ls_term = k2_sq_exp_double_dev_2_grad_ls(
                                qn2, qn1, q1, q2n2, sig, ls
                            )
                            ls_derv += ls_term * dq1_dr * dqn2_dr * fi * fnj

                if c1 == s and c2 == s:
                    for i in range(len(neigh2_array_1)):
                        q2n1 = neigh2_array_1[i, s1]
                        en1 = neigh_etypes1[m][i]
                        rni = neigh_bond_array_1[m][i, 0]
                        fni, _ = cutoff_func(r_cut, rni, 0)

                        for j in range(len(neigh2_array_2)):
                            q2n2 = neigh2_array_2[j, s2]
                            en2 = neigh_etypes2[n][j]
                            rnj = neigh_bond_array_2[n][j, 0]
                            fnj, _ = cutoff_func(r_cut, rnj, 0)

                            if e1 == e2 and en1 == en2:
                                dk_dqn1qn2 = k2_sq_exp_double_dev_1(
                                    qn1, qn2, q2n1, q2n2, sig, ls
                                )
                                kern += (
                                    dk_dqn1qn2 * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2
                                ls_term = k2_sq_exp_double_dev_1_grad_ls(
                                    qn1, qn2, q2n1, q2n2, sig, ls
                                )
                                ls_derv += (
                                    ls_term * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2

                            if e1 == en2 and en1 == e2:
                                dk_dqn1qn2 = k2_sq_exp_double_dev_2(
                                    qn1, q2n2, qn2, q2n1, sig, ls
                                )
                                kern += (
                                    dk_dqn1qn2 * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2
                                ls_term = k2_sq_exp_double_dev_2_grad_ls(
                                    qn1, q2n2, qn2, q2n1, sig, ls
                                )
                                ls_derv += (
                                    ls_term * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2
    sig_derv = 2 / sig * kern

    return kern, np.array([sig_derv, ls_derv])


@njit
def many_body_mc_jit(
    q_array_1,
    q_array_2,
    q_neigh_array_1,
    q_neigh_array_2,
    bond_array_1,
    bond_array_2,
    q_neigh_grads_1,
    q_neigh_grads_2,
    neigh_bond_array_1,
    neigh_bond_array_2,
    q_neigh2_array_1,
    q_neigh2_array_2,
    c1,
    c2,
    etypes1,
    etypes2,
    neigh_etypes1,
    neigh_etypes2,
    species1,
    species2,
    sig,
    ls,
    d1,
    d2,
    r_cut,
    cutoff_func,
):
    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8
    )
    kern = 0

    for s in useful_species:
        s1 = np.where(species1 == s)[0][0]
        s2 = np.where(species2 == s)[0][0]

        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        # this has been calculated in env_getarray.py
        dq1_dr = 0.0
        for m in range(q_neigh_array_1.shape[0]):
            e1 = etypes1[m]
            if e1 == s:
                dq1_dr += q_neigh_grads_1[m, d1 - 1]

        dq2_dr = 0.0
        for n in range(q_neigh_array_2.shape[0]):
            e2 = etypes2[n]
            if e2 == s:
                dq2_dr += q_neigh_grads_2[n, d2 - 1]

        for m in range(q_neigh_array_1.shape[0]):
            qn1 = q_neigh_array_1[m, s1]
            ri = bond_array_1[m, 0]
            ci = bond_array_1[m, d1]
            fi, dfi = cutoff_func(r_cut, ri, ci)
            e1 = etypes1[m]

            for n in range(q_neigh_array_2.shape[0]):
                qn2 = q_neigh_array_2[n, s2]
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, dfj = cutoff_func(r_cut, rj, cj)
                e2 = etypes2[n]

                if c1 == c2 and e1 == e2:
                    kq = k2_sq_exp(q1, q2, qn1, qn2, sig, ls)
                    kern += kq * dfj * dfi  # 0-0

                    dk_dq1 = k2_sq_exp_dev(q1, q2, qn1, qn2, sig, ls)
                    kern += dq1_dr * dk_dq1 * fi * dfj  # 1-0

                    dk_dq2 = k2_sq_exp_dev(q2, q1, qn2, qn1, sig, ls)
                    kern += dq2_dr * dk_dq2 * dfi * fj  # 0-1

                    dk_dq1q2 = k2_sq_exp_double_dev_1(q1, q2, qn1, qn2, sig, ls)  # 1-1
                    kern += dk_dq1q2 * dq1_dr * dq2_dr * fi * fj

                if c1 == e2 and c2 == e1:
                    kq = k2_sq_exp(q1, qn2, qn1, q2, sig, ls)
                    kern += kq * dfj * dfi

                    dk_dq1 = k2_sq_exp_dev(q1, qn2, qn1, q2, sig, ls)
                    kern += dq1_dr * dk_dq1 * fi * dfj

                    dk_dq2 = k2_sq_exp_dev(q2, qn1, q1, qn2, sig, ls)
                    kern += dq2_dr * dk_dq2 * dfi * fj

                    dk_dq1q2 = k2_sq_exp_double_dev_2(q1, qn2, q2, qn1, sig, ls)
                    kern += dk_dq1q2 * dq1_dr * dq2_dr * fi * fj

                # 2nd neighbors
                if c1 == s:
                    dqn1_dr = q_neigh_grads_1[m, d1 - 1]
                    neigh2_array_1 = q_neigh2_array_1[m]
                    for i in range(len(neigh2_array_1)):
                        q2n1 = neigh2_array_1[i, s1]
                        en1 = neigh_etypes1[m][i]
                        rni = neigh_bond_array_1[m][i, 0]
                        fni, _ = cutoff_func(r_cut, rni, 0)

                        if e1 == c2 and en1 == e2:
                            dk_dq = k2_sq_exp_dev(qn1, q2, q2n1, qn2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * dfj  # 2-0

                            dk_dqn1q2 = k2_sq_exp_double_dev_1(
                                qn1, q2, q2n1, qn2, sig, ls
                            )
                            kern += dk_dqn1q2 * dqn1_dr * dq2_dr * fni * fj  # 2-1

                        if en1 == c2 and e1 == e2:
                            dk_dq = k2_sq_exp_dev(qn1, qn2, q2n1, q2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * dfj

                            dk_dqn1q2 = k2_sq_exp_double_dev_2(
                                qn1, qn2, q2, q2n1, sig, ls
                            )
                            kern += dk_dqn1q2 * dqn1_dr * dq2_dr * fni * fj

                # 2nd neighbors
                if c2 == s:
                    dqn2_dr = q_neigh_grads_2[n, d2 - 1]
                    neigh2_array_2 = q_neigh2_array_2[n]
                    for j in range(len(neigh2_array_2)):
                        q2n2 = neigh2_array_2[j, s2]
                        en2 = neigh_etypes2[n][j]
                        rnj = neigh_bond_array_2[n][j, 0]
                        fnj, _ = cutoff_func(r_cut, rnj, 0)

                        if e2 == c1 and en2 == e1:
                            dk_dq = k2_sq_exp_dev(qn2, q1, q2n2, qn1, sig, ls)
                            kern += dk_dq * dqn2_dr * dfi * fnj  # 0-2

                            dk_dq1qn2 = k2_sq_exp_double_dev_1(
                                qn2, q1, q2n2, qn1, sig, ls
                            )
                            kern += dk_dq1qn2 * dq1_dr * dqn2_dr * fi * fnj  # 1-2

                        if en2 == c1 and e2 == e1:
                            dk_dq = k2_sq_exp_dev(qn2, qn1, q2n2, q1, sig, ls)
                            kern += dk_dq * dqn2_dr * dfi * fnj

                            dk_dq1qn2 = k2_sq_exp_double_dev_2(
                                qn2, qn1, q1, q2n2, sig, ls
                            )
                            kern += dk_dq1qn2 * dq1_dr * dqn2_dr * fi * fnj

                if c1 == s and c2 == s:
                    for i in range(len(neigh2_array_1)):
                        q2n1 = neigh2_array_1[i, s1]
                        en1 = neigh_etypes1[m][i]
                        rni = neigh_bond_array_1[m][i, 0]
                        fni, _ = cutoff_func(r_cut, rni, 0)

                        for j in range(len(neigh2_array_2)):
                            q2n2 = neigh2_array_2[j, s2]
                            en2 = neigh_etypes2[n][j]
                            rnj = neigh_bond_array_2[n][j, 0]
                            fnj, _ = cutoff_func(r_cut, rnj, 0)

                            if e1 == e2 and en1 == en2:
                                dk_dqn1qn2 = k2_sq_exp_double_dev_1(
                                    qn1, qn2, q2n1, q2n2, sig, ls
                                )
                                kern += (
                                    dk_dqn1qn2 * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2
                            if e1 == en2 and en1 == e2:
                                dk_dqn1qn2 = k2_sq_exp_double_dev_2(
                                    qn1, q2n2, qn2, q2n1, sig, ls
                                )
                                kern += (
                                    dk_dqn1qn2 * dqn1_dr * dqn2_dr * fni * fnj
                                )  # 2-2

    return kern


@njit
def many_body_mc_force_en_jit(
    q_array_1,
    q_array_2,
    q_neigh_array_1,
    q_neigh_array_2,
    bond_array_1,
    bond_array_2,
    q_neigh_grads_1,
    neigh_bond_array_1,
    q_neigh2_array_1,
    c1,
    c2,
    etypes1,
    etypes2,
    neigh_etypes1,
    species1,
    species2,
    sig,
    ls,
    d1,
    r_cut,
    cutoff_func,
):
    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8
    )
    kern = 0

    for s in useful_species:
        s1 = np.where(species1 == s)[0][0]
        s2 = np.where(species2 == s)[0][0]

        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        # this has been calculated in env_getarray.py
        dq1_dr = 0.0
        for m in range(q_neigh_array_1.shape[0]):
            e1 = etypes1[m]
            if e1 == s:
                dq1_dr += q_neigh_grads_1[m, d1 - 1]

        for n in range(q_neigh_array_2.shape[0]):
            qn2 = q_neigh_array_2[n, s2]
            rj = bond_array_2[n, 0]
            fj, _ = cutoff_func(r_cut, rj, 0)
            e2 = etypes2[n]

            for m in range(q_neigh_array_1.shape[0]):
                qn1 = q_neigh_array_1[m, s1]
                ri = bond_array_1[m, 0]
                ci = bond_array_1[m, d1]
                fi, dfi = cutoff_func(r_cut, ri, ci)
                e1 = etypes1[m]

                if c1 == c2 and e1 == e2:
                    kq = k2_sq_exp(q1, q2, qn1, qn2, sig, ls)
                    kern += kq * fj * dfi
                    dk_dq = k2_sq_exp_dev(q1, q2, qn1, qn2, sig, ls)
                    kern += dq1_dr * dk_dq * fi * fj

                if c1 == e2 and c2 == e1:
                    kq = k2_sq_exp(q1, qn2, qn1, q2, sig, ls)
                    kern += kq * fj * dfi
                    dk_dq = k2_sq_exp_dev(q1, qn2, qn1, q2, sig, ls)
                    kern += dq1_dr * dk_dq * fi * fj

                # 2nd neighbors
                dqn1_dr = q_neigh_grads_1[m, d1 - 1]
                neigh2_array = q_neigh2_array_1[m]
                for i in range(len(neigh2_array)):
                    q2n1 = neigh2_array[i, s1]
                    en1 = neigh_etypes1[m][i]
                    rni = neigh_bond_array_1[m][i, 0]
                    fni, _ = cutoff_func(r_cut, rni, 0)

                    if e1 == c2 and en1 == e2:
                        if c1 == s:
                            dk_dq = k2_sq_exp_dev(qn1, q2, q2n1, qn2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * fj
                    if en1 == c2 and e1 == e2:
                        if c1 == s:
                            dk_dq = k2_sq_exp_dev(qn1, qn2, q2n1, q2, sig, ls)
                            kern += dk_dq * dqn1_dr * fni * fj

    kern /= -2

    return kern


@njit
def many_body_mc_en_jit(
    q_array_1,
    q_array_2,
    q_neigh_array_1,
    q_neigh_array_2,
    bond_array_1,
    bond_array_2,
    c1,
    c2,
    etypes1,
    etypes2,
    species1,
    species2,
    sig,
    ls,
    r_cut,
    cutoff_func,
):
    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8
    )
    kern = 0

    for s in useful_species:  # To be improved: set different species for q1, qn1
        s1 = np.where(species1 == s)[0][0]
        s2 = np.where(species2 == s)[0][0]

        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        for m in range(q_neigh_array_1.shape[0]):
            qn1 = q_neigh_array_1[m, s1]
            ri = bond_array_1[m, 0]
            fi, _ = cutoff_func(r_cut, ri, 0)
            e1 = etypes1[m]

            for n in range(q_neigh_array_2.shape[0]):
                qn2 = q_neigh_array_2[n, s2]
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                e2 = etypes2[n]

                if c1 == c2 and e1 == e2:
                    kern += k2_sq_exp(q1, q2, qn1, qn2, sig, ls) * fi * fj

                if c1 == e2 and c2 == e1:
                    kern += k2_sq_exp(q1, qn2, qn1, q2, sig, ls) * fi * fj

    kern /= 4

    return kern
