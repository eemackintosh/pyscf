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

from pyscf.cc import gqcisd_slow as gqcisd


def UQCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_ghf(mf)
    return gqcisd.GQCISD(mf, 2*frozen, mo_coeff, mo_occ)


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

    mycc = UQCISD(mf, frozen=2)
    ecc, t1, t2 = mycc.kernel()
    print(mycc.e_tot - -149.976883)
    et = mycc.qcisd_t()
    print(mycc.e_tot+et)
    print(mycc.e_tot+et - -149.986265)
