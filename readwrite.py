from subproblem import *
from masterproblem import *
import os
#Updated read-write file
def rw(i, al_x1, al_x2, al_x3, al_la, al_mu, be, max_hw, scomp, gcostc1, gcostc2, gcostc3, rho, tps, \
       stc1, mut1, mdt1, stc2, mut2, mdt2, stc3, mut3, mdt3):
    c="alpha_x1.txt"
    f = open(c, "a")
    for j in range(1, tps+1):
        f.write(str(i))
        f.write(" ")
        f.write(str(j))
        f.write(" ")
        f.write(str(al_x1[j-1]))
        f.write("\n")
    f.close()
    c = "alpha_x2.txt"
    f = open(c, "a")
    for j in range(1, tps+1):
        f.write(str(i))
        f.write(" ")
        f.write(str(j))
        f.write(" ")
        f.write(str(al_x2[j-1]))
        f.write("\n")
    f.close()
    c = "alpha_x3.txt"
    f = open(c, "a")
    for j in range(1, tps+1):
        f.write(str(i))
        f.write(" ")
        f.write(str(j))
        f.write(" ")
        f.write(str(al_x3[j-1]))
        f.write("\n")
    f.close()

    c = "alpha_la.txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(al_la)+"\n")
    f.close()
    c = "alpha_mu.txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(al_mu)+"\n")
    f.close()
    c = "betas.txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(be))
    f.write("\n")
    f.close()
    data = "param ccount:="
    data += str(i)
    data += ";\n"
    data += "param max_hw:="
    data += str(max_hw)
    data += ";\n"
    #First stage costs are now passed from the main file
    data += "param gcostc1:="
    data += str(gcostc1)
    data += ";\n"
    data += "param gcostc2:="
    data += str(gcostc2)
    data += ";\n"
    data += "param gcostc3:="
    data += str(gcostc3)
    data += ";\n"
    #Rho is now passed from the main file
    data += "param rho:="
    data += str(rho)
    data += ";\n"
    data += "param scomp:="
    data += str(scomp)
    data += ";\n"
    data += "param pn:="
    data += str(tps)
    data += ";\n"
    data += "param stc1:="
    data += str(stc1)
    data += ";\n"
    data += "param mut1:="
    data += str(mut1)
    data += ";\n"
    data += "param mdt1:="
    data += str(mdt1)
    data += ";\n"
    data += "param stc2:="
    data += str(stc2)
    data += ";\n"
    data += "param mut2:="
    data += str(mut2)
    data += ";\n"
    data += "param mdt2:="
    data += str(mdt2)
    data += ";\n"
    data += "param stc3:="
    data += str(stc3)
    data += ";\n"
    data += "param mut3:="
    data += str(mut3)
    data += ";\n"
    data += "param mdt3:="
    data += str(mdt3)
    data += ";\n"
    data11 = data12= data2 = data3 = data4 = ""
    c = "alpha_x1.txt"
    with open(c) as fp:
        data11 = fp.read()
    c = "alpha_x2.txt"
    with open(c) as fp:
        data12 = fp.read()
    c = "alpha_x3.txt"
    with open(c) as fp:
        data13 = fp.read()
    c = "alpha_la.txt"
    with open(c) as fp:
        data2 = fp.read()
    c = "alpha_mu.txt"
    with open(c) as fp:
        data3 = fp.read()
    c = "betas.txt"
    with open(c) as fp:
        data4 = fp.read()

    data += "param alpha_x1:=\n"
    data += data11
    data += ";\n\n"
    data += "param alpha_x2:=\n"
    data += data12
    data += ";\n\n"
    data += "param alpha_x3:=\n"
    data += data13
    data += ";\n\n"
    data += "param alpha_la:=\n"
    data += data2
    data += ";\n\n"
    data += "param alpha_mu:=\n"
    data += data3
    data += ";\n\n"
    data += "param beta:=\n"
    data += data4
    data += ";\n"
    c = "master1.dat"
    with open(c, 'w') as fp:
        fp.write(data)