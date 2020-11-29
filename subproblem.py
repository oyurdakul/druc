from pyomo.environ import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory
import numpy as np
from scipy.stats import bernoulli


import logging
#
def subp_sub(x1_in, x2_in, x3_in, nd, tps, gcostl1, gcostl2, gcostl3, bp, glim1, \
             glim2, glim3, pramp1, pramp2, pramp3, nramp1, nramp2, nramp3, gcostc1, gcostc2, gcostc3):
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)

    opt = SolverFactory('gurobi', solver_io="python")

    model1=AbstractModel()

    subproblem=model1.create_instance()
    subproblem.pernum = RangeSet(tps)
    subproblem.pernum2=RangeSet(2,tps)
    subproblem.gen1 = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.gen2 = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.gen3 = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.buy = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.sell=Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.x1int = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.x2int = Var(subproblem.pernum, within=NonNegativeReals)
    subproblem.x3int = Var(subproblem.pernum, within=NonNegativeReals)

    # Power balance based on bought and generated power
    def c1func(subproblem, i):
        return (-subproblem.gen1[i]-subproblem.gen2[i]-subproblem.gen3[i]) - subproblem.buy[i] + nd[i - 1] + subproblem.sell[i]  == 0

    # Power generation limits
    def c2func1(subproblem, i):
        return subproblem.gen1[i] - (subproblem.x1int[i] * glim1) <= 0

    def c2func2(subproblem, i):
        return subproblem.gen2[i] - (subproblem.x2int[i] * glim2) <= 0

    def c2func3(subproblem, i):
        return subproblem.gen3[i] - (subproblem.x3int[i] * glim3) <= 0

    def c3func1(subproblem, i):
        return subproblem.x1int[i] == x1_in[i-1]

    def c3func2(subproblem, i):
        return subproblem.x2int[i] == x2_in[i-1]

    def c3func3(subproblem, i):
        return subproblem.x3int[i] == x3_in[i-1]
    def c4func1(subproblem, i):
        return (subproblem.gen1[i]-subproblem.gen1[i-1]) <= pramp1

    def c5func1(subproblem, i):
        return (subproblem.gen1[i]-subproblem.gen1[i-1]) >= nramp1

    def c4func2(subproblem, i):
        return (subproblem.gen2[i]-subproblem.gen2[i-1]) <= pramp2

    def c5func2(subproblem, i):
        return (subproblem.gen2[i]-subproblem.gen2[i-1]) >= nramp2

    def c4func3(subproblem, i):
        return (subproblem.gen3[i]-subproblem.gen3[i-1]) <= pramp3

    def c5func3(subproblem, i):
        return (subproblem.gen3[i]-subproblem.gen3[i-1]) >= nramp3

    def objfnc(subproblem):
        sum = 0
        for i in subproblem.pernum:
            sum += ((gcostl1 * subproblem.gen1[i]) + \
                    (gcostl2 * subproblem.gen2[i]) + \
                    (gcostl3 * subproblem.gen3[i])  + \
                     (bp[i - 1] * subproblem.buy[i]))
        return sum


    subproblem.c1 = Constraint(subproblem.pernum, rule=c1func)
    subproblem.c21 = Constraint(subproblem.pernum, rule=c2func1)
    subproblem.c22 = Constraint(subproblem.pernum, rule=c2func2)
    subproblem.c23 = Constraint(subproblem.pernum, rule=c2func3)
    subproblem.c31 = Constraint(subproblem.pernum, rule=c3func1)
    subproblem.c32 = Constraint(subproblem.pernum, rule=c3func2)
    subproblem.c33 = Constraint(subproblem.pernum, rule=c3func3)
    subproblem.c41 = Constraint(subproblem.pernum2, rule=c4func1)
    subproblem.c51 = Constraint(subproblem.pernum2, rule=c5func1)
    subproblem.c42 = Constraint(subproblem.pernum2, rule=c4func2)
    subproblem.c52 = Constraint(subproblem.pernum2, rule=c5func2)
    subproblem.c43 = Constraint(subproblem.pernum2, rule=c4func3)
    subproblem.c53 = Constraint(subproblem.pernum2, rule=c5func3)

    subproblem.obj = Objective(rule=objfnc)

    subproblem.dual = Suffix(direction=Suffix.IMPORT)
    opt.solve(subproblem)

    hwx=value(subproblem.obj)
    optgen1= [0]

    for i in subproblem.pernum:
        optgen1 = np.concatenate((optgen1, np.array([value(subproblem.gen1[i])])), axis=0)

    optgen1 = np.delete(optgen1, [0])

    optgen2= [0]

    for i in subproblem.pernum:
        optgen2 = np.concatenate((optgen2, np.array([value(subproblem.gen2[i])])), axis=0)

    optgen2 = np.delete(optgen2, [0])

    optgen3= [0]

    for i in subproblem.pernum:
        optgen3 = np.concatenate((optgen3, np.array([value(subproblem.gen3[i])])), axis=0)

    optgen3 = np.delete(optgen3, [0])

    optbuy = [0]

    for i in subproblem.pernum:
        optbuy = np.concatenate((optbuy, np.array([value(subproblem.buy[i])])), axis=0)

    optbuy = np.delete(optbuy, [0])

    optsell = [0]

    for i in subproblem.pernum:
        optsell = np.concatenate((optsell, np.array([value(subproblem.sell[i])])), axis=0)

    optsell = np.delete(optsell, [0])

    pi_one =  np.zeros(tps)
    pi_two_1 = np.zeros(tps)
    pi_two_2 = np.zeros(tps)
    pi_two_3 = np.zeros(tps)
    pi_three_1 = np.zeros(tps)
    pi_three_2 = np.zeros(tps)
    pi_three_3 = np.zeros(tps)

    for i in range(tps):
        pi_one[i]=value(subproblem.dual[subproblem.c1[i+1]])
        pi_two_1[i] = value(subproblem.dual[subproblem.c21[i+1]])
        pi_two_2[i] = value(subproblem.dual[subproblem.c22[i + 1]])
        pi_two_3[i] = value(subproblem.dual[subproblem.c23[i + 1]])
        pi_three_1[i] = value(subproblem.dual[subproblem.c31[i + 1]])
        pi_three_2[i] = value(subproblem.dual[subproblem.c32[i + 1]])
        pi_three_3[i] = value(subproblem.dual[subproblem.c33[i + 1]])

    return (hwx, optgen1, optgen2, optgen3, optbuy, optsell, pi_one, pi_two_1, pi_two_2, pi_two_3, pi_three_1, pi_three_2, pi_three_3)

def cuts(x1_in, x2_in, x3_in, l_in, m_in, hwxs, pi_1_s, pi_21_s, pi_22_s,pi_23_s, pi_31_s, pi_32_s,pi_33_s, sprob, sns,\
         tps, glim1, glim2, glim3, gcostc1, gcostc2, gcostc3):
    s_agg = np.zeros(sns)
    ubs= np.zeros(sns)
    al_x1_agg = np.zeros((sns, tps))
    al_x2_agg = np.zeros((sns, tps))
    al_x3_agg = np.zeros((sns, tps))
    al_l_agg = np.zeros(sns)
    al_m_agg = np.zeros(sns)
    be_agg = np.zeros(sns)
    for i in range(sns):
        s_agg[i] = (hwxs[i] - m_in) / l_in
        ubs[i]= l_in * exp(s_agg[i] - 1) #
        al_l_agg[i] = exp(s_agg[i] - 1) - (s_agg[i] * exp(s_agg[i] - 1))
        al_m_agg[i] = -exp(s_agg[i] - 1)
        for j in range(tps):
            al_x1_agg[i,j]=exp(s_agg[i]-1)*(pi_31_s[i, j])
            al_x2_agg[i,j]=exp(s_agg[i]-1)*(pi_32_s[i, j])
            al_x3_agg[i, j]=exp(s_agg[i]-1)*(pi_33_s[i, j])
        xsum = 0
        for j in range(tps):
            xsum += al_x1_agg[i,j]*x1_in[j]
            xsum += al_x2_agg[i, j] * x2_in[j]
            xsum += al_x3_agg[i, j] * x3_in[j]
        be_agg[i]=ubs[i] - (xsum + al_l_agg[i]*l_in + al_m_agg[i]*m_in)

    al_l = al_m = be = 0
    al_x1 = np.zeros(tps)
    al_x2 = np.zeros(tps)
    al_x3 = np.zeros(tps)
    for j in range(sns):
        al_l += (al_l_agg[j] * sprob[j])
        al_m += (al_m_agg[j] * sprob[j])
        be += (be_agg[j] * sprob[j])
    for i in range(tps):
        for j in range(sns):
            al_x1[i] += al_x1_agg[j,i] * sprob[j]
            al_x2[i] += al_x2_agg[j, i] * sprob[j]
            al_x3[i] += al_x3_agg[j, i] * sprob[j]
    xsum = 0
    for i in range(tps):
        xsum += (al_x1[i] * x1_in[i] + al_x2[i] * x2_in[i] + al_x3[i] * x3_in[i] )
    ub = xsum + (al_l * l_in) + (al_m * m_in) + be
    return (al_x1, al_x2, al_x3, al_l, al_m, be, ub)

def subp(a, bpr, scaling, x1_in, x2_in, x3_in, l_in, m_in, tps, gcostl1, gcostl2, gcostl3, glim1, glim2, glim3, \
         pramp1, pramp2, pramp3, nramp1, nramp2, nramp3, gcostc1, gcostc2, gcostc3):

    model=ConcreteModel()
    file1 = open(a, 'r')
    Lines = file1.readlines()
    sns = len(Lines) #number of scenarios

    model.sn=RangeSet(sns)
    model.tp=RangeSet(tps)

    model.sprob=Param(model.sn, mutable=True)
    model.nl = Param(model.sn, model.tp, mutable=True)
    model.bp = Param(model.sn, model.tp, mutable=True)
    model.pi1 = Param(model.sn, model.tp, mutable=True)
    model.pi21 = Param(model.sn, model.tp, mutable=True)
    model.pi22 = Param(model.sn, model.tp, mutable=True)
    model.pi23 = Param(model.sn, model.tp, mutable=True)
    model.pi31 = Param(model.sn, model.tp, mutable=True)
    model.pi32 = Param(model.sn, model.tp, mutable=True)
    model.pi33 = Param(model.sn, model.tp, mutable=True)
    model.hwx=Param(model.sn, mutable=True)
    model.gen1_agg=Param(model.sn, model.tp, mutable=True)
    model.gen2_agg = Param(model.sn, model.tp, mutable=True)
    model.gen3_agg = Param(model.sn, model.tp, mutable=True)
    model.buy_agg = Param(model.sn, model.tp, mutable=True)
    model.sell_agg=Param(model.sn, model.tp, mutable=True)
    model.al_x1_agg = Param(model.sn, model.tp, mutable=True)
    model.al_x2_agg = Param(model.sn, model.tp, mutable=True)
    model.al_x3_agg = Param(model.sn, model.tp, mutable=True)
    model.al_l = Param(model.sn, mutable=True)
    model.al_m = Param(model.sn, mutable=True)
    model.be = Param(model.sn, mutable=True)

    j = 1
    for line in Lines:
        senas=line.split(",")
        for i in range(tps):
            model.nl[j, i+1]=float(senas[i])/scaling
            model.bp[j, i+1] = bpr
        model.sprob[j] = float(senas[24])
        j += 1

    nload = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.nl[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        nload = np.concatenate((nload, [temp]), axis=0)
    nload = np.delete(nload, 0, axis=0)
    hwx = sprob = [0]
    for i in model.sn:
        sprob =  np.concatenate((sprob, np.array([value(model.sprob[i])])), axis=0)
    sprob = np.delete(sprob, [0])
    for j in model.sn:
        netload = [0]
        for i in model.tp:
            netload = np.concatenate((netload, np.array([value(model.nl[j, i])])), axis=0)
        netload = np.delete(netload, [0])
        bp = [0]
        for i in model.tp:
            bp = np.concatenate((bp, np.array([value(model.bp[j, i])])), axis=0)
        bp = np.delete(bp, [0])
        hwx_t, gens1, gens2, gens3, buys, sells, pi1, pi2_1, pi2_2, pi2_3, pi3_1, pi3_2, pi3_3 = subp_sub(x1_in, x2_in, x3_in, netload, tps, gcostl1, \
                                                      gcostl2, gcostl3, bp, glim1, glim2, glim3, pramp1, \
                                                      pramp2, pramp3, nramp1, nramp2, nramp3, gcostc1, gcostc2, gcostc3)
        hwx = np.concatenate((hwx, np.array([hwx_t])), axis=0)
        for i in model.tp:
            model.pi1[j, i] = pi1[i - 1]
            model.pi21[j, i] = pi2_1[i - 1]
            model.pi22[j, i] = pi2_2[i - 1]
            model.pi23[j, i] = pi2_3[i - 1]
            model.pi31[j, i] = pi3_1[i - 1]
            model.pi32[j, i] = pi3_2[i - 1]
            model.pi33[j, i] = pi3_3[i - 1]
            model.gen1_agg[j, i]=gens1[i-1]
            model.gen2_agg[j, i] = gens2[i - 1]
            model.gen3_agg[j, i] = gens3[i - 1]
            model.buy_agg[j, i] = buys[i-1]
            model.sell_agg[j,i]=sells[i-1]
    hwx = np.delete(hwx, [0])
    pi_1_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi1[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_1_a = np.concatenate((pi_1_a, [temp]), axis=0)
    pi_1_a = np.delete(pi_1_a, 0, axis=0)

    pi_21_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi21[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_21_a = np.concatenate((pi_21_a, [temp]), axis=0)
    pi_21_a = np.delete(pi_21_a, 0, axis=0)

    pi_22_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi22[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_22_a = np.concatenate((pi_22_a, [temp]), axis=0)
    pi_22_a = np.delete(pi_22_a, 0, axis=0)

    pi_23_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi23[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_23_a = np.concatenate((pi_23_a, [temp]), axis=0)
    pi_23_a = np.delete(pi_23_a, 0, axis=0)

    pi_31_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi31[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_31_a = np.concatenate((pi_31_a, [temp]), axis=0)
    pi_31_a = np.delete(pi_31_a, 0, axis=0)

    pi_32_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi32[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_32_a = np.concatenate((pi_32_a, [temp]), axis=0)
    pi_32_a = np.delete(pi_32_a, 0, axis=0)

    pi_33_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.pi33[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        pi_33_a = np.concatenate((pi_33_a, [temp]), axis=0)
    pi_33_a = np.delete(pi_33_a, 0, axis=0)

    gen1_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.gen1_agg[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        gen1_a = np.concatenate((gen1_a, [temp]), axis=0)
    gen1_a = np.delete(gen1_a, 0, axis=0)
    # for i in range(sns):
    #     for j in range(tps):
    #         print(gen1_a[i,j])

    gen2_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.gen2_agg[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        gen2_a = np.concatenate((gen2_a, [temp]), axis=0)
    gen2_a = np.delete(gen2_a, 0, axis=0)

    gen3_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.gen3_agg[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        gen3_a = np.concatenate((gen3_a, [temp]), axis=0)
    gen3_a = np.delete(gen3_a, 0, axis=0)

    buy_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.buy_agg[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        buy_a = np.concatenate((buy_a, [temp]), axis=0)
    buy_a = np.delete(buy_a, 0, axis=0)

    sell_a = np.zeros((1, tps))
    for i in range(sns):
        temp = [0]
        for j in range(tps):
            temp = np.concatenate((temp, np.array([value(model.sell_agg[i + 1, j + 1])])), axis=0)
        temp = np.delete(temp, [0])
        sell_a = np.concatenate((sell_a, [temp]), axis=0)
    sell_a = np.delete(sell_a, 0, axis=0)

    maxim = hwx[0]
    for j in range(sns):
        if value(hwx[j]) > maxim:
            maxim = value(hwx[j])

    al_x1, al_x2, al_x3, al_la, al_mu, be, ub = cuts(x1_in, x2_in, x3_in, l_in, m_in, hwx, pi_1_a, \
                                       pi_21_a, pi_22_a, pi_23_a, pi_31_a, pi_32_a, pi_33_a, sprob, sns, tps, glim1, \
                                                     glim2, glim3, gcostc1, gcostc2, gcostc3)
    # for i in range(tps):
    #     print("calculated alpha for x1 in time period", i, ": ", al_x1[i])
    # for i in range(tps):
    #     print("calculated alpha for x2 in time period", i, ": ", al_x2[i])
    # for i in range(tps):
    #     print("calculated alpha for x3 in time period", i, ": ", al_x3[i])
    # print("calculated alpha for lambda", al_la)
    # print("calculated alpha for mu", al_mu)
    # print("calculated beta", be)
    print("upper bound: ", ub)
    return (maxim, ub, sns, gen1_a, gen2_a, gen3_a, buy_a, sell_a, al_x1, al_x2, al_x3, al_la, al_mu, be, nload)