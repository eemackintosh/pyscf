#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
Generalized QCISD (spin-orbital) implementation
The 4-index integrals are saved on disk entirely (without using any symmetry).

Note MO integrals are treated in physicist's notation

Ref: 
'''

import numpy
import numpy as np

from pyscf import lib
from pyscf.cc import gccsd as gccsd
from pyscf.cc import qcisd_slow as qcisd


def update_amps(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Fvv = fvv - 0.5*lib.einsum('mnaf,mnef->ae', t2, eris.oovv)
    Foo = foo + 0.5*lib.einsum('inef,mnef->mi', t2, eris.oovv)
    Fov = fov + lib.einsum('nf,mnef->me', t1, eris.oovv)

    Woooo = eris.oooo \
            + 0.25*lib.einsum('ijef,mnef->mnij', t2, eris.oovv)
    Wvvvv = np.asarray(eris.vvvv) \
            + 0.25*lib.einsum('mnab,mnef->abef', t2, eris.oovv)
    Wovvo = -np.asarray(eris.ovov).transpose(0,1,3,2) \
            - 0.5*lib.einsum('jnfb,mnef->mbej', t2, eris.oovv)

    Fvv -= np.diag(np.diag(fvv))
    Foo -= np.diag(np.diag(foo))

    t1new = np.array(fov).conj()
    t1new +=  lib.einsum('ie,ae->ia', t1, Fvv)
    t1new += -lib.einsum('ma,mi->ia', t1, Foo)
    t1new +=  lib.einsum('imae,me->ia', t2, Fov)
    t1new += -lib.einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*lib.einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*lib.einsum('mnae,mnie->ia', t2, eris.ooov)

    t2new = np.array(eris.oovv).conj()
    tmp = lib.einsum('ijae,be->ijab', t2, Fvv)
    t2new += (tmp - tmp.transpose(0,1,3,2))
    tmp = lib.einsum('imab,mj->ijab', t2, Foo)
    t2new -= (tmp - tmp.transpose(1,0,2,3))
    t2new += 0.5*lib.einsum('mnab,mnij->ijab', t2, Woooo)
    t2new += 0.5*lib.einsum('ijef,abef->ijab', t2, Wvvvv)
    tmp = lib.einsum('imae,mbej->ijab', t2, Wovvo)
    t2new += (tmp - tmp.transpose(0,1,3,2)
              - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2) )
    tmp = lib.einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    #tmp = lib.einsum('ma,mbij->ijab', t1, eris.ovoo)
    tmp = lib.einsum('ma,ijmb->ijab', t1, np.array(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:] - cc.level_shift
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    e = 0.25*lib.einsum('ijab,ijab', t2, eris.oovv)
    return e.real


class GQCISD(gccsd.GCCSD):

    def kernel(self, t1=None, t2=None, eris=None):
        return self.qcisd(t1, t2, eris)
    def qcisd(self, t1=None, t2=None, eris=None):
        return qcisd.QCISD.qcisd(self, t1, t2, eris)

    update_amps = update_amps
    energy = energy

    def qcisd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import gqcisd_t_slow as qcisd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return qcisd_t.kernel(self, eris, t1, t2)


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = """O 0 0 0
                  O 0 0 1.2169"""
    mol.basis = 'cc-pvdz'
    mol.verbose = 7
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    mf = scf.addons.convert_to_ghf(mf)

    mycc = GQCISD(mf, frozen=4).run()
    print(mycc.e_tot - -149.976883)
    et = mycc.qcisd_t()
    print(mycc.e_tot+et)
    print(mycc.e_tot+et - -149.986265)
