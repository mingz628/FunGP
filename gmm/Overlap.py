import math

import numpy as np
from numpy.linalg import eig, det, eigh
from mpmath import mp
import sys

class Overlap:
    def __init__(self, p, k, pi, mu, s, pars, lim):
        self.sigsq = None
        self.lmax = None
        self.lmin = None
        self.mean = None
        self.c = 1.0
        self.ndtsrt = None
        self.r = None
        self.fail = None
        self.intl = None
        self.ersm = None
        self.lb = None
        self.nc = None
        self.n = None
        self.p = p
        self.k = k
        self.pi = pi
        self.mu = mu
        self.s = s
        self.pars = pars
        self.lim = lim
        self.ifault = None
        self.li = np.zeros((self.k, self.k, self.p))
        self.di = np.zeros((self.k, self.k, self.p))
        self.const1 = np.zeros((self.k, self.k))
        self.omega_map = np.eye(self.k)
        self.asympt = 0
        self.compute_pars()
        self.get_omega_map()

    def compute_pars(self):
        dets = np.zeros(self.k)
        sh = np.zeros((self.k, self.p, self.p))
        sinv = np.zeros((self.k, self.p, self.p))
        for i in range(self.k):
            ga = self.s[i]
            val, vec = eigh(ga)
            determinant = np.prod(val)
            vec *= (-1)
            dets[i] = determinant

            l = np.diag(np.float_power(val, 0.5))
            sh[i] = np.matmul(np.matmul(vec, l), np.transpose(vec))

            l = np.diag(np.float_power(val, -1))
            sinv[i] = np.matmul(np.matmul(vec, l), np.transpose(vec))

        m1 = np.zeros(self.p)
        for i in range(self.k - 1):
            for j in range(i + 1, self.k):
                for l in range(self.p):
                    m1[l] = self.mu[i][l] - self.mu[j][l]
                m2 = m1 * -1

                si = np.matmul(np.matmul(sh[i], sinv[j]), np.transpose(sh[i]))
                val, vec = eigh(si)
                vec *= (-1)
                self.li[i][j] = val
                ga = np.matmul(sinv[i], sh[i])
                val = ga.dot(m1)
                ga = np.transpose(vec)
                m1 = ga.dot(val)
                self.di[i][j] = m1

                si = np.matmul(np.matmul(sh[j], sinv[i]), np.transpose(sh[j]))
                val, vec = eigh(si)
                vec *= (-1)
                self.li[j][i] = val
                ga = np.matmul(sinv[j], sh[j])
                val = ga.dot(m2)
                ga = np.transpose(vec)
                m2 = ga.dot(val)
                self.di[j][i] = m2

                self.const1[i][j] = mp.log(mp.fmul(self.pi[j], self.pi[j]) / mp.fmul(self.pi[i], self.pi[i]) * dets[i] / dets[j])
                self.const1[j][i] = -self.const1[i][j]


    def get_omega_map(self):
        self.acc = self.pars[1]
        total_omega = 0.0
        max_omega = 0.0
        df = np.ones(self.p)
        i = 0
        j = 1
        hom = 1
        for l in range(self.p):
            if self.li[0][1][l] != self.li[1][0][l]:
                hom = 0

        if hom == 1:
            while i < self.k - 1:
                Di = self.di[i][j] / (self.c ** 0.5)
                cnst1 = np.sum(np.square(Di))
                coef = np.zeros(self.p)
                ncp = np.zeros(self.p)
                self.t = self.const1[i][j] - cnst1
                self.sigma = 2 * (np.float_power(cnst1, 0.5))
                self.omega_map[i][j] = self.qfc(coef, ncp, df, self.p, self.sigma, self.t, self.lim, self.acc, self.ifault)

                Di = self.di[i][j] / (self.c ** 0.5)
                cnst1 = np.sum(np.square(Di))
                coef = np.zeros(self.p)
                self.t = self.const1[j][i] - cnst1
                self.sigma = 2 * (np.float_power(cnst1, 0.5))
                self.omega_map[j][i] = self.qfc(coef, ncp, df, self.p, self.sigma, self.t, self.lim, self.acc,self.ifault)
                if j < self.k-1:
                    j += 1
                else:
                    i += 1
                    j = i + 1

        if hom == 0:
            self.sigma = 0.0
            while i < self.k - 1:
                Di = self.di[i][j] / (self.c ** 0.5)
                cnst1 = self.const1[i][j]
                coef = self.li[i][j] - 1.0
                ldprod = self.li[i][j] * Di
                const2 = ldprod * Di / coef
                s = np.sum(const2)
                ncp = np.square(ldprod / coef)
                self.t = s + cnst1
                self.omega_map[i][j] = self.qfc(coef, ncp, df, self.p, self.sigma, self.t, self.lim, self.acc, self.ifault)

                Di = self.di[j][i] / (self.c ** 0.5)
                cnst1 = self.const1[j][i]
                coef = self.li[j][i] - 1.0
                ldprod = self.li[j][i] * Di
                const2 = ldprod * Di / coef
                s = np.sum(const2)
                ncp = np.square(ldprod / coef)
                self.t = s + cnst1
                self.omega_map[j][i] = self.qfc(coef, ncp, df, self.p, self.sigma, self.t, self.lim, self.acc, self.ifault)
                if j < (self.k - 1):
                    j += 1
                else:
                    i += 1
                    j = i + 1
        for l in range(self.k):
            self.omega_map[l][l] = 0.0

    def exp1(self, x):
        return 0.0 if x < -50.0 else mp.exp(x)

    def log1(self, x, first):
        x = math.fabs(x)
        if x > 0.11:
            return math.log(1.0 + x) if first else (math.log(1.0 + x) - x)
        else:
            y = x / (2.0 + x)
            term = 2.0 * (y ** 3)
            k = 3.0
            s = 2.0 * y if first else -x * y
            y = y ** 2
            s1 = s + term / k
            while s1 != s:
                k = k + 2.0
                term = term * y
                s = s1
                s1 = s + term / k
            return s

    def truncation(self, u, tausq):
        sum1, prod2, prod3, s = 0, 0, 0, 0
        sum2 = (self.sigsq + tausq) * u ** 2
        prod1 = 2.0 * sum2
        u *= 2.0
        for lj, ncj, nj in zip(self.lb, self.nc, self.n):
            x = (u * lj) ** 2
            sum1 += ncj * x / (1.0 + x)
            if x > 1:
                prod2 += nj * math.log(x)
                prod3 = prod3 + nj * self.log1(x, True)
                s = s + nj
            else:
                prod1 += nj * self.log1(x, True)
        sum1 *= 0.5
        prod2 = prod1 + prod2
        prod3 = prod1 + prod3
        x = self.exp1(-sum1 - 0.25 * prod2) / math.pi
        y = self.exp1(-sum1 - 0.25 * prod3) / math.pi
        err1 = 1.0 if s == 0 else x * 2.0 / s
        err2 = 2.5 * y if prod3 > 1.0 else 1.0
        if err2 < err1:
            err1 = err2
        x = 0.5 * sum2
        err2 = 1.0 if x <= y else y / x
        return err1 if err1 < err2 else err2

    def order(self, th):
        k = 0
        for j in range(self.r):
            lj = math.fabs(self.lb[j])
            swapped = True
            for k in range(j - 1, -1, -1):
                if lj > math.fabs(self.lb[th[k]]):
                    th[k + 1] = th[k]
                else:
                    swapped = False
                    break
            if swapped:
                k = -1
            th[k + 1] = j
        self.ndtsrt = False
        return th

    def cfe(self, x, th):
        if self.ndtsrt:
            th = self.order(th)
        axl = math.fabs(x)
        sxl = 1.0 if x > 0.0 else -1.0
        sum1 = 0.0
        for j in range(self.r - 1, -1, -1):
            t = th[j]
            if (self.lb[t] * sxl) > 0.0:
                lj = math.fabs(self.lb[t])
                axl1 = axl - lj * (self.n[t] + self.nc[t])
                axl2 = lj / .0866
                if axl1 > axl2:
                    axl = axl1
                else:
                    if axl > axl2:
                        axl = axl2
                    sum1 = (axl - axl1) / lj
                    for k in range(j - 1, -1, -1):
                        sum1 += self.n[th[k]] + self.nc[th[k]]
                    break
        if sum1 > 100.0:
            self.fail = True
            return 1.0, th
        else:
            return 2.0 ** (sum1 / 4.0) / (math.pi * axl ** 2), th

    def find_u(self, utx, accx):
        divis = np.array([2.0, 1.4, 1.2, 1.1])
        ut = utx
        u = ut / 4.0
        if self.truncation(u, 0.0) > accx:
            u = ut
            while self.truncation(u, 0.0) > accx:
                ut *= 4.0
                u = ut
        else:
            u /= 4.0
            while self.truncation(u, 0.0) <= accx:
                u /= 4.0
            ut = u
        for i in range(4):
            u = ut / divis[i]
            if self.truncation(u, 0.0) <= accx:
                ut = u
        return ut

    def errbd(self, u):
        xconst = u * self.sigsq
        sum1 = u * xconst
        u1 = 2.0 * u
        for j in range(self.r - 1, -1, -1):
            nj = self.n[j]
            lj = self.lb[j]
            ncj = self.nc[j]
            x = mp.fmul(u1, lj)
            y =  1.0 - x if (1.0 - x) != 0 else sys.float_info.min
            mul1 = mp.fmul(lj, (ncj / y + nj))
            div1 = mp.fdiv(mul1, y)
            xconst += div1
            sum1 += ncj * (x / y) ** 2 + nj * (x ** 2 / y + self.log1(-x, False))
        return self.exp1(-0.5 * sum1), xconst

    def ctff(self, accx, upn):
        u2 = upn
        u1 = 0.0
        c1 = self.mean
        rb = 2.0 * self.lmax if u2 > 0.0 else 2 * self.lmin
        u = u2 / (1.0 + u2 * rb)
        err_result, c2 = self.errbd(u)
        while err_result > accx:
            u1 = u2
            c1 = c2
            u2 *= 2.0
            u = u2 / (1.0 + u2 * rb)
            err_result, c2 = self.errbd(u)
        divisor = (c2 - self.mean) if abs(c2 - self.mean) > abs(sys.float_info.min) else sys.float_info.min
        u = (c1 - self.mean) / divisor
        while u < 0.9:
            u = (u1 + u2) / 2.0
            err_result, xconst = self.errbd(u / (1.0 + u * rb))
            if err_result > accx:
                u1 = u
                c1 = xconst
            else:
                u2 = u
                c2 = xconst
            divisor = (c2 - self.mean) if abs(c2 - self.mean) > abs(sys.float_info.min) else sys.float_info.min
            u = (c1 - self.mean) / divisor
            if (c1 - self.mean) == 0 and (c2 - self.mean) == 0:
                break
        return c2, u2

    def integrate(self, nterm, interv, tausq, mainx):
        inpi = interv / math.pi
        for k in range(nterm, -1, -1):
            u = (k + 0.5) * interv
            sum1 = - 2.0 * u * self.c1
            sum2 = math.fabs(sum1)
            sum3 = - 0.5 * self.sigsq * u ** 2
            for j in range(self.r - 1, -1, -1):
                nj = self.n[j]
                x = 2.0 * self.lb[j] * u
                y = x ** 2
                sum3 = sum3 - 0.25 * nj * self.log1(y, True)
                y = self.nc[j] * x / (1.0 + y)
                z = nj * math.atan(x) + y
                sum1 = sum1 + z
                sum2 = sum2 + math.fabs(z)
                sum3 = sum3 - 0.5 * x * y
            x = inpi * self.exp1(sum3) / u
            if not mainx:
                x *= (1.0 - self.exp1(-0.5 * tausq * u ** 2))
            sum1 = math.sin(0.5 * sum1) * x
            sum2 = 0.5 * sum2 * x
            self.intl += sum1
            self.ersm += sum2

    def qfc(self, lb1, nc1, n1, r1in, sigmain, c1in, lim1in, accin, ifault):
        r1, lim1 = r1in, lim1in
        sigma, c1, acc = sigmain, c1in, accin
        qfval = 0
        rats = np.array([1, 2, 4, 8])
        self.r, lim, self.c1 = r1, lim1, c1
        self.n, self.lb, self.nc = n1, lb1, nc1
        ifault, count, self.intl, self.ersm, qfval, acc1, self.ndtsrt, self.fail, xlim = 0, 0, 0.0, 0.0, -1.0, acc, True, False, float(
            lim)
        th = np.array([0] * self.r)
        self.sigsq = sigma ** 2

        # Potentially use loop to find all three
        sd = self.sigsq + np.sum(np.float_power(self.lb, 2) * (2 * self.n + 4 * self.nc))
        self.mean = np.sum(self.lb * (self.n + self.nc))
        self.lmax, self.lmin = np.max(self.lb), np.min(self.lb)

        if sd == 0:
            return 1.0 if sd == 0.0 else 0.0
        if self.lmin == 0.0 and self.lmax == 0.0 and sigma == 0.0:
            raise Exception("Invalid parameters")
        sd = math.sqrt(sd)
        almx = max(-self.lmin, self.lmax)
        utx = 16.0 / sd
        up = 4.5 / sd
        un = - up
        utx = self.find_u(utx, .5 * acc1)
        if self.t != 0.0 and (almx > 0.07 * sd):
            cfe_result, th = self.cfe(self.c1, th)
            tausq = .25 * acc1 / cfe_result
            if self.fail:
                self.fail = False
            elif self.truncation(utx, tausq) < .2 * acc1:
                self.sigsq = self.sigsq + tausq
                utx = self.find_u(utx, .25 * acc1)
        acc1 = 0.5 * acc1
        while True:
            ctff_result, up = self.ctff(acc1, up)
            d1 = ctff_result - self.c1
            if d1 < 0.0:
                qfval = 1.0
                return qfval
            ctff_result, un = self.ctff(acc1, un)
            d2 = self.c1 - ctff_result
            if d2 < 0.0:
                qfval = 0.0
                return qfval
            intv = 2.0 * math.pi / max(d1, d2)
            xnt = utx / intv
            xntm = 3.0 / math.sqrt(acc1)
            if xnt > xntm * 1.5:
                if xntm > xlim:
                    raise Exception('required accuracy NOT achieved')
                ntm = int(math.floor(xntm + 0.5))
                intv1 = utx / ntm
                x = 2.0 * math.pi / intv1
                if x <= math.fabs(self.c1):
                    break
                cfe_result1, th = self.cfe(self.c1 - x, th)
                cfe_result2, th = self.cfe(self.c1 + x, th)
                tausq = .33 * acc1 / (1.1 * (cfe_result1 + cfe_result2))
                if self.fail:
                    break
                acc1 = .67 * acc1
                self.integrate(ntm, intv1, tausq, False)
                xlim -= xntm
                self.sigsq += tausq
                utx = self.find_u(utx, .25 * acc1)
                acc1 *= 0.75
            else:
                break
        if xnt > xlim:
            raise Exception('required accuracy NOT achieved')
        nt = int(math.floor(xnt + 0.5)) if not math.isnan(xnt) else -1
        self.integrate(nt, intv, 0.0, True)
        qfval = 0.5 - self.intl

        up = self.ersm
        x = up + acc / 10.0
        for j in range(4):
            if rats[j] * x == rats[j] * up:
                ifault = 2
        return qfval