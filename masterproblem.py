from pyomo.environ import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory
import numpy as np
from scipy.stats import bernoulli

import logging


def mpr(mulimit):
    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    opt = SolverFactory('gurobi')
    model2 = AbstractModel()
    model2.ccount = Param() #number of cuts created
    model2.pn = Param()
    model2.pnum = RangeSet(model2.pn)
    model2.tset = RangeSet(model2.ccount) #RangeSet for the number of cuts
    model2.alpha_x1 = Param(model2.tset, model2.pnum)
    model2.alpha_x2 = Param(model2.tset, model2.pnum)
    model2.alpha_x3 = Param(model2.tset, model2.pnum)
    model2.alpha_la = Param(model2.tset)
    model2.alpha_mu = Param(model2.tset)
    model2.beta = Param(model2.tset)
    model2.stc1=Param()
    model2.mut1 = Param()
    model2.mdt1 = Param()
    model2.stc2=Param()
    model2.mut2 = Param()
    model2.mdt2 = Param()
    model2.stc3=Param()
    model2.mut3 = Param()
    model2.mdt3 = Param()
    model2.x1 = Var(model2.pnum, within=Binary)
    model2.start_up1 = Var(model2.pnum, within=Binary)
    model2.x2 = Var(model2.pnum, within=Binary)
    model2.start_up2 = Var(model2.pnum, within=Binary)
    model2.x3 = Var(model2.pnum, within=Binary)
    model2.start_up3 = Var(model2.pnum, within=Binary)

    model2.la = Var(within=PositiveReals)
    model2.mu = Var()
    model2.theta = Var(within=Reals)

    model2.max_hw=Param()
    model2.gcostc1=Param()
    model2.gcostc2 = Param()
    model2.gcostc3 = Param()
    model2.rho=Param()

    def objfnc(model2):
        sum = 0
        for j in model2.pnum:
            sum += model2.gcostc1*model2.x1[j]
            sum += model2.start_up1[j] * value(model2.stc1)
            sum += model2.gcostc2 * model2.x2[j]
            sum += model2.start_up2[j] * value(model2.stc2)
            sum += model2.gcostc3 * model2.x3[j]
            sum += model2.start_up3[j] * value(model2.stc3)
        return sum + model2.mu + (model2.rho * model2.la) + model2.theta

    def c2func(model2, i):
        sumx = 0
        for j in model2.pnum:
            sumx += model2.alpha_x1[i,j]*model2.x1[j]
            sumx += model2.alpha_x2[i, j] * model2.x2[j]
            sumx += model2.alpha_x3[i, j] * model2.x3[j]
        return model2.theta >= (sumx + model2.alpha_la[i] * model2.la + \
                                model2.alpha_mu[i] * model2.mu + model2.beta[i])

    def c10func1(model2, j):
        if j == 1:
            return model2.start_up1[j] == model2.x1[j]
        else:
            return model2.start_up1[j] >= model2.x1[j] - model2.x1[j - 1]


    # up-time constraint

    def c11func1(model2, j):
        sum = 0
        for i in range(j, j + value(model2.mut1)):
            if i <= value(model2.pn):
                sum += model2.x1[i]
        return sum >= value(model2.mut1) * model2.start_up1[j]


    # down-time constraint

    def c12func1(model2, j):
        sum = 0
        if j == 1:
            for i in range(j, j + value(model2.mdt1)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x1[i]
            return sum >= value(model2.mdt1) * (0 - model2.x1[j])
        else:
            for i in range(j, j + value(model2.mdt1)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x1[i]
            return sum >= value(model2.mdt1) * (model2.x1[j - 1] - model2.x1[j])

    def c10func2(model2, j):
        if j == 1:
            return model2.start_up2[j] == model2.x2[j]
        else:
            return model2.start_up2[j] >= model2.x2[j] - model2.x2[j - 1]


    # up-time constraint


    def c11func2(model2, j):
        sum = 0
        for i in range(j, j + value(model2.mut2)):
            if i <= value(model2.pn):
                sum += model2.x2[i]
        return sum >= value(model2.mut2) * model2.start_up2[j]


    # down-time constraint


    def c12func2(model2, j):
        sum = 0
        if j == 1:
            for i in range(j, j + value(model2.mdt2)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x2[i]
            return sum >= value(model2.mdt2) * (0 - model2.x2[j])
        else:
            for i in range(j, j + value(model2.mdt2)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x2[i]
            return sum >= value(model2.mdt2) * (model2.x2[j - 1] - model2.x2[j])


    def c10func3(model2, j):
        if j == 1:
            return model2.start_up3[j] == model2.x3[j]
        else:
            return model2.start_up3[j] >= model2.x3[j] - model2.x3[j - 1]


    # up-time constraint


    def c11func3(model2, j):
        sum = 0
        for i in range(j, j + value(model2.mut3)):
            if i <= value(model2.pn):
                sum += model2.x3[i]
        return sum >= value(model2.mut3) * model2.start_up3[j]


    # down-time constraint


    def c12func3(model2, j):
        sum = 0
        if j == 1:
            for i in range(j, j + value(model2.mdt3)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x3[i]
            return sum >= value(model2.mdt3) * (0 - model2.x3[j])
        else:
            for i in range(j, j + value(model2.mdt3)):
                if i <= value(model2.pn):
                    sum += 1 - model2.x3[i]
            return sum >= value(model2.mdt3) * (model2.x3[j - 1] - model2.x3[j])

    model2.scomp=Param()
    model2.c2 = Constraint(model2.tset, rule=c2func)
    model2.c3 = Constraint(expr=model2.la>=0.0001)
    model2.c4 = Constraint(expr=model2.mu >=mulimit)
    model2.c5 = Constraint(expr=model2.mu <= 180000)
    model2.c6 = Constraint(expr=model2.mu >= model2.max_hw - (model2.la * model2.scomp))
    model2.c101 = Constraint(model2.pnum, rule=c10func1)
    model2.c111 = Constraint(model2.pnum, rule=c11func1)
    model2.c121 = Constraint(model2.pnum, rule=c12func1)
    model2.c102 = Constraint(model2.pnum, rule=c10func2)
    model2.c112 = Constraint(model2.pnum, rule=c11func2)
    model2.c122 = Constraint(model2.pnum, rule=c12func2)
    model2.c103 = Constraint(model2.pnum, rule=c10func3)
    model2.c113 = Constraint(model2.pnum, rule=c11func3)
    model2.c123 = Constraint(model2.pnum, rule=c12func3)

    model2.obj = Objective(rule=objfnc)

    c = "master1.dat"
    mproblem = model2.create_instance(c)
    def c8fnc1(mproblem, i):
        return mproblem.x1[i]==0
    def c8fnc2(mproblem, i):
        return mproblem.x2[i]==0
    def c8fnc3(mproblem, i):
        return mproblem.x3[i]==0

    if value(mproblem.ccount)==1:
        mproblem.c7 = Constraint(expr=mproblem.la==1)
        mproblem.c81 = Constraint(mproblem.pnum, rule=c8fnc1)
        mproblem.c82 = Constraint(mproblem.pnum, rule=c8fnc2)
        mproblem.c83 = Constraint(mproblem.pnum, rule=c8fnc3)




    opt.solve(mproblem, warmstart=True)
    # print("master problem output")
    # print("calculated x", value(mproblem.x))
    # print("calculated lambda", value(mproblem.la))
    # print("iteration", value(mproblem.ccount),  "calculated mu", value(mproblem.mu),\
    #       "calculated lambda", value(mproblem.la))
    # for i in mproblem.pnum:
    #     print("first generator commitment for time period", i, ": ", value(mproblem.x1[i]))
    # for i in mproblem.pnum:
    #     print("second generator commitment for time period", i, ": ", value(mproblem.x2[i]))
    # for i in mproblem.pnum:
    #     print("third generator commitment for time period", i, ": ", value(mproblem.x3[i]))
    print("calculated lower bound", value(mproblem.theta))
    # print("calculated objective function value", value(mproblem.obj) )
    # print("non first-stage costs: ", value(mproblem.obj) - value(mproblem.x))
    sval=(value(mproblem.max_hw)-value(mproblem.mu)) / value(mproblem.la)
    # print("sval in iteration ",value(mproblem.ccount), ": ", sval)
    x1vals = np.zeros(value(mproblem.pn))
    x2vals = np.zeros(value(mproblem.pn))
    x3vals = np.zeros(value(mproblem.pn))
    for i in mproblem.pnum:
        x1vals[i - 1] = value(mproblem.x1[i])
        x2vals[i - 1] = value(mproblem.x2[i])
        x3vals[i - 1] = value(mproblem.x3[i])
    return (x1vals, x2vals, x3vals, value(mproblem.la), value(mproblem.mu), value(mproblem.theta), value(mproblem.obj))