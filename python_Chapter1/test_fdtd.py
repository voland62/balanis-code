"""Numerical checks: scalar Maxwell updates, source, PEC and absorption."""
import unittest

import numpy as np

from fdtd import Simulation


class FDTDTests(unittest.TestCase):
    def test_scalar_tm_updates(self):
        # Independent scalar loops catch shifted slices, signs and update ordering.
        rng = np.random.default_rng(42)
        for scenario in ("tm_box", "tm_open"):
            sim = Simulation(scenario)
            for array in sim.fields.values():
                array[:] = rng.normal(size=array.shape)
            old = {k: v.copy() for k, v in sim.fields.items()}
            expected = {k: v.copy() for k, v in old.items()}
            s = sim.size
            for i in range(1, s):
                for j in range(1, s):
                    xd = old["hy"][i, j] - old["hy"][i - 1, j]
                    yd = old["hx"][i, j - 1] - old["hx"][i, j]
                    if sim.npml:
                        for name, diff in (("ezx", xd), ("ezy", yd)):
                            a, b = sim.coefficients[name]
                            expected[name][i, j] = a[i, j] * old[name][i, j] + b[i, j] * diff
                        expected["ez"][i, j] = expected["ezx"][i, j] + expected["ezy"][i, j]
                    else:
                        expected["ez"][i, j] += sim.dt / sim.eps / sim.dx * (xd + yd)
            u = 4 / 25 / sim.dt * (sim.dt - 25 * sim.dt)
            expected["ez"][sim.ic, sim.jc] = 2 * u * np.exp(-u*u) / sim.dx
            for i in range(s + 1):
                for j in range(s):
                    a, b = sim.coefficients["hx"]
                    expected["hx"][i, j] = a[i, j]*old["hx"][i, j] + b[i, j]*(expected["ez"][i, j]-expected["ez"][i, j+1])
            for i in range(s):
                for j in range(s + 1):
                    a, b = sim.coefficients["hy"]
                    expected["hy"][i, j] = a[i, j]*old["hy"][i, j] + b[i, j]*(expected["ez"][i+1, j]-expected["ez"][i, j])
            sim.step()
            for name in expected:
                np.testing.assert_allclose(sim.fields[name], expected[name], rtol=1e-14, atol=1e-14)

    def test_scalar_te_updates(self):
        sim = Simulation("te_horn")
        rng = np.random.default_rng(43)
        for array in sim.fields.values():
            array[:] = rng.normal(size=array.shape)
        old = {k: v.copy() for k, v in sim.fields.items()}
        expected = {k: v.copy() for k, v in old.items()}
        s = sim.size
        for i in range(s):
            for j in range(1, s):
                a, b = sim.coefficients["ex"]
                expected["ex"][i, j] = a[i, j]*old["ex"][i, j] + b[i, j]*(old["hz"][i, j]-old["hz"][i, j-1])
        for i in range(1, s):
            for j in range(s):
                a, b = sim.coefficients["ey"]
                expected["ey"][i, j] = a[i, j]*old["ey"][i, j] + b[i, j]*(old["hz"][i-1, j]-old["hz"][i, j])
        source = (1-np.exp(-(1/20)**2))*sim.impedance*np.sin(2*np.pi*sim.frequency*sim.dt)
        expected["ey"][sim.ic-10, sim.jc-2:sim.jc+2] = source
        for i in range(s):
            for j in range(s):
                for name, diff in (("hzx", expected["ey"][i, j]-expected["ey"][i+1, j]),
                                   ("hzy", expected["ex"][i, j+1]-expected["ex"][i, j])):
                    a, b = sim.coefficients[name]
                    expected[name][i, j] = a[i, j]*old[name][i, j]+b[i, j]*diff
                expected["hz"][i, j] = expected["hzx"][i, j]+expected["hzy"][i, j]
        sim.step()
        for name in expected:
            np.testing.assert_allclose(sim.fields[name], expected[name], rtol=1e-14, atol=1e-14)

    def test_original_geometry_and_pml_indices(self):
        sim = Simulation("te_horn")
        i, j = sim.ic, sim.jc
        # Explicit MATLAB (one-based) wall locations, converted independently.
        ex_walls = {(ii-1, jj-1) for ii in range(17, 33) for jj in (48, 52)}
        for ii, lo, hi in ((33,47,53),(34,47,53),(35,46,54),(36,46,54),
                           (37,45,55),(38,45,55),(39,44,56),(40,44,56),(41,43,57)):
            ex_walls.update(((ii-1, lo-1), (ii-1, hi-1)))
        ey_walls = {(16, jj-1) for jj in range(48,52)}
        for ii, lo, hi in ((33,47,52),(35,46,53),(37,45,54),(39,44,55),(41,43,56)):
            ey_walls.update(((ii-1, lo-1), (ii-1, hi-1)))
        for name, walls in (("ex",ex_walls),("ey",ey_walls)):
            a, b = sim.coefficients[name]
            self.assertEqual(set(map(tuple, np.argwhere(b == 0))), walls)
            self.assertTrue(all(a[index] == -1 for index in walls))
        for scenario in ("tm_open", "te_horn"):
            sim = Simulation(scenario)
            p, s = sim.npml, sim.size
            sigmax = -3*sim.eps*sim.c*np.log(1e-5)/(2*sim.dx*p)
            for name, axis, magnetic in (("ey" if sim.te else "ezx",0,False),
                                         ("ex" if sim.te else "ezy",1,False),
                                         ("hzx" if sim.te else "hy",0,True),
                                         ("hzy" if sim.te else "hx",1,True)):
                a, b = sim.coefficients[name]
                for k in range(1,p+1):
                    loss = (sigmax*sim.impedance**2*(k/(p+.5))**2 if magnetic
                            else sigmax*((k-.5)/(p+.5))**2)
                    constant = sim.mu if magnetic else sim.eps
                    aa = np.exp(-loss*sim.dt/constant)
                    bb = -(aa-1)/loss/sim.dx
                    for position in ((p-k if magnetic else p+1-k), s-p+k-1):
                        index = (position,s//2) if axis == 0 else (s//2,position)
                        self.assertAlmostEqual(a[index], aa)
                        self.assertAlmostEqual(b[index], bb)

    def test_symmetry_pec_and_absorption(self):
        box, opened = Simulation("tm_box"), Simulation("tm_open")
        for _ in range(210):
            box.step()
            opened.step()
            np.testing.assert_allclose(box.field, box.field.T, atol=1e-10)
            np.testing.assert_allclose(box.field, box.field[::-1], atol=1e-10)
        for sim in (box, opened):
            self.assertTrue(np.isfinite(sim.field).all())
            self.assertTrue(np.all(sim.field[[0,-1],:] == 0))
            self.assertTrue(np.all(sim.field[:,[0,-1]] == 0))
        # Compare the physical interior after the outgoing pulse has left it.
        core = np.s_[8:-8,8:-8]
        self.assertLess(np.linalg.norm(opened.field[core]), .1*np.linalg.norm(box.field[core]))
        horn = Simulation("te_horn")
        for _ in range(210):
            horn.step()
        for name in ("ex", "ey"):
            walls = horn.coefficients[name][1] == 0
            self.assertTrue(np.all(horn.fields[name][walls] == 0))
        self.assertTrue(all(np.isfinite(a).all() for a in horn.fields.values()))


if __name__ == "__main__":
    unittest.main()
