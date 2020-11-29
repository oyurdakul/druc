from subproblem import *
from masterproblem import *
from readwrite import *
import time as tm
import numpy as np
import os
st=tm.time()
dislis=['euclidean']
d=0
while(d<len(dislis)):
    dis=dislis[d]
    d += 1
    #monlis=['1','2','4','6','8','10','12','14','16','18','20','22','24','26','28']
    monlis = ['12']
    m=0
    while(m<len(monlis)):
        mon=monlis[m]
        m+=1
        rholis=[0]
        r=0
        while(r<len(rholis)):
            rho=rholis[r]
            r+=1


            ######
            scomp =15;
            mulimit = -500000
            tps =24
            tol=0.0001
            bpr = 22
            scaling = 2500
            #######136215.26988073057


            al_x1=np.zeros(tps); al_x2=np.zeros(tps); al_x3=np.zeros(tps); al_la=0; al_mu=0; be=-50000; \
                ub=100000; lb=-50000;  \
                gcostc1=115/scaling; glim1=591/scaling; gcostl1=10; pramp1=591/scaling; nramp1=-591/scaling;\
                stc1 = 0/scaling; mut1 = 1; mdt1 = 1;\
                gcostc2=150/scaling; glim2=215/scaling; gcostl2=15; pramp2=215/scaling; nramp2=-215/scaling;\
                stc2 = 0/scaling; mut2 = 1; mdt2 = 1;\
                gcostc3 =180/scaling; glim3=192/scaling; gcostl3 = 20; pramp3 = 192/scaling; nramp3 = -192/scaling; \
                stc3 = 0/scaling; mut3 = 1; mdt3 = 1;

            ##
                # gcostc1 = 50 / scaling; glim1 = 591 / scaling; gcostl1 = 10; pramp1 = 210 / scaling; nramp1 = -210/scaling; \
                # stc1 = 65 scaling; mut1 = 7; mdt1 = 4; \
                # gcostc2 = 65 / scaling; glim2 = 215 / scaling; gcostl2 = 15;pramp2 = 125 / scaling; nramp2 = -125 / scaling; \
                # stc2 = 85 / scaling; mut2 = 4; mdt2 = 2; \
                # gcostc3 = 80 / scaling; glim3 = 192 / scaling; gcostl3 = 20; pramp3 = 50 / scaling; nramp3 = -50 / scaling; \
                # stc3 = 100 / scaling; mut3 = 1; mdt3 = 1;
            ##
            snns=12

            al_x1 = np.zeros(tps);
            al_x2 = np.zeros(tps);
            al_x3 = np.zeros(tps);
            al_la = 0;
            al_mu = 0;
            be = -50000; \
            ub = 100000;
            lb = -50000;
            max_hw = 25000;

            i = 1
            seninp = 'j_scen/scenarios/new/'+str(dis)+'_'+str(mon)+'.txt';
            while (abs(ub-lb)>tol):
                print(i)
                et = tm.time()
                dur = et - st
                print('duration: ', dur)
                rw(i,al_x1,al_x2, al_x3, al_la, al_mu, be, max_hw, scomp, gcostc1, gcostc2, gcostc3,  rho, tps, stc1, mut1, mdt1,\
                stc2, mut2, mdt2, stc3, mut3, mdt3)
                f_x1, f_x2, f_x3, f_la, f_mu, lb, obj = mpr(mulimit)
                # print(f_x1)
                # print(f_x2)
                # print(f_x3)
                maxim, ub, sn, gens1, gens2, gens3, buys, sells, al_x1, al_x2, al_x3, al_la, al_mu, be, netload = subp(seninp, \
                bpr, scaling, f_x1,f_x2,f_x3, f_la, f_mu, tps, gcostl1, gcostl2, gcostl3, glim1, glim2, glim3, pramp1, pramp2, pramp3,\
                nramp1, nramp2, nramp3, gcostc1, gcostc2, gcostc3)
                if i==1:
                    max_hw=maxim
                i += 1




            for i in range(tps):
                for j in range(sn):
                    print("generator 1 generation in time period", i+1, "scenario", j+1, gens1[j,i]*scaling)
            for i in range(tps):
                for j in range(sn):
                    print("generator 2 generation in time period", i+1, "scenario", j+1, gens2[j,i]*scaling)
            for i in range(tps):
                for j in range(sn):
                    print("generator 3 generation in time period", i+1, "scenario", j+1, gens3[j,i]*scaling)

            for i in range(tps):
                for j in range(sn):
                    print("purchased power in time period", i+1, "scenario", j+1, buys[j,i]*scaling)

            for i in range(tps):
                for j in range(sn):
                    print("curtailed power in time period", i+1, "scenario", j+1, sells[j,i]*scaling)

            for i in range(tps):
                print("x1 commitment in time period ",i+1, f_x1[i])
            for i in range(tps):
                print("x2 commitment in time period ", i + 1, f_x2[i])
            for i in range(tps):
                print("x3 commitment in time period ", i + 1, f_x3[i])

            print("la value", f_la)
            print("mu value", f_mu)

            print("objective function value: ", obj * scaling)


            f = open("j_scen/j_results/"+str(tps)+".txt", "w")
            f.write("x1=[")
            for i in range(tps-1):
                f.write(str(f_x1[i]))
                f.write(",")
            f.write(str(f_x1[tps-1]))
            f.write("]\n")
            f.write("x2=[")
            for i in range(tps-1):
                f.write(str(f_x2[i]))
                f.write(",")
            f.write(str(f_x2[tps-1]))
            f.write("]\n")
            f.write("x3=[")
            for i in range(tps-1):
                f.write(str(f_x3[i]))
                f.write(",")
            f.write(str(f_x3[tps-1]))
            f.write("]\n")
            f.write("obj: ")
            f.write(str(obj*scaling))
            f.close()

            os.remove("alpha_x1.txt")
            os.remove("alpha_x2.txt")
            os.remove("alpha_x3.txt")
            os.remove("alpha_la.txt")
            os.remove("alpha_mu.txt")
            os.remove("betas.txt")
            os.remove("master1.dat")


            os.system('afplay /System/Library/Sounds/Sosumi.aiff')