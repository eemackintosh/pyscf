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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
GHF-QCISD(T) with spin-orbital integrals
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import gccsd

# spin-orbital formula
# JCP 98, 8718 (1993); DOI:10.1063/1.464480
def kernel(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    assert(isinstance(eris, gccsd._PhysicistsERIs))
    if t1 is None or t2 is None:
        t1, t2 = cc.t1, cc.t2

    nocc, nvir = t1.shape
    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)

    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    t3c =(numpy.einsum('jkae,bcei->ijkabc', t2, bcei) -
          numpy.einsum('imbc,majk->ijkabc', t2, majk))
    t3c = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
    t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
    t3c /= d3
    # e4 = numpy.einsum('ijkabc,ijkabc,ijkabc', t3c.conj(), d3, t3c) / 36
    # sia = numpy.einsum('jkbc,ijkabc->ia', eris.oovv, t3c) * .25
    # e5 = numpy.einsum('ia,ia', sia, t1.conj())
    # et = e4 + e5
    # return et
    t3d = numpy.einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += numpy.einsum('ai,jkbc->ijkabc', eris.fock[nocc:,:nocc], t2)
    t3d = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
    t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
    t3d /= d3
    et = numpy.einsum('ijkabc,ijkabc,ijkabc', (t3c+2*t3d).conj(), d3, t3c) / 36
    return et
