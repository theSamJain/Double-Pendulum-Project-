import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import animationchaos as chanim

def func(y,x,consts=None):

    #Constants
    g, m1, m2, l1, l2 = consts

    # initial value 1, initial value 2, ... = y     #initial y 
    th1, o1, th2, o2 = y

    #Equations
    d_o1 = ( -g * (2*m1 + m2) * np.sin(th1) - m2 * g * np.sin(th1 - 2 * th2) - 2 * np.sin(th1 - th2) * m2 * (o2 ** 2 * l2 + o1 ** 2 * l1 * np.cos(th1 - th2)) ) / ( l1 * (2 * m1 + m2 - m2 * np.cos(2 * (th1 - th2))) )
    d_o2 = ( 2 * np.sin(th1 - th2)*(o1 ** 2 * l1 * (m1 + m2) + (m1 + m2) * g * np.cos(th1) + o2 ** 2 * l2 * m2 * np.cos(th1 - th2)) ) / ( l2 * (2 * m1 + m2 - m2 * np.cos(2 * (th1 - th2))) )

    # dydt = [First Eqn, Second Eqn, ...]
    dydt = [o1, d_o1, o2, d_o2]

    return dydt
  
def rk4(y0, x, consts):

    h = (x[-1] - x[0]) / len(x)
    yarr = []
    yin = y0
    xin = [x[0]]

    for i in range(len(x)):
        yarr.append(yin)

        k1 = [h * ele for ele in func(yin, xin, consts)]

        yn = [e1 + e2/2 for (e1, e2) in zip(yin, k1)]
        xn = [e1 + h/2 for e1 in xin]
        k2 = [h * ele for ele in func(yn, xn, consts)]

        yn = [e1 + e2/2 for (e1, e2) in zip(yin, k2)]
        k3 = [h * ele for ele in func(yn, xn, consts)]

        yn = [e1 + e2 for (e1, e2) in zip(yin, k3)]
        xn = [e1 + h for e1 in xin]
        k4= [h * ele for ele in func(yn, xn, consts)]

        yf = [ini_y + (e1 + 2 * (e2 + e3) + e4) / 6 for (ini_y,e1,e2,e3,e4) in zip(yin, k1, k2, k3, k4)]

        yin = yf
        xin = [e1 + h/2 for e1 in xn]

    yarr=np.array(yarr).reshape(-1,4)

    return(yarr)

def track(sol_RK, folname):
    x1 = np.sin(sol_RK[:, 0])*consts[3]
    x2 = x1 + np.sin(sol_RK[:, 2])*consts[4]
    y1 = -1 * np.cos(sol_RK[:, 0])*consts[3]
    y2 = y1 - np.cos(sol_RK[:, 2])*consts[4]
    data = np.column_stack((sol_RK[:, 0], sol_RK[:, 1], sol_RK[:, 2], sol_RK[:, 3], t))
    coordt = np.column_stack((x1, y1, x2, y2, t))
    np.savetxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\data.txt'%folname, data, header = "theta 1     Omega 1     Theta 2     Omega 2      Time")
    np.savetxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\coordinates.txt'%folname, coordt, header = "x1       y1      x2      y2     Time")

    return (x1, y1, x2, y2)

def addtrack(sol_RK2, folname):
    a1 = np.sin(sol_RK2[:, 0])*consts[3]
    a2 = a1 + np.sin(sol_RK2[:, 2])*consts[4]
    b1 = -1 * np.cos(sol_RK2[:, 0])*consts[3]
    b2 = b1 - np.cos(sol_RK2[:, 2])*consts[4]
    data2 = np.column_stack((sol_RK2[:, 0], sol_RK2[:, 1], sol_RK2[:, 2], sol_RK2[:, 3], t ))
    coordt2 = np.column_stack((a1, b1, a2, b2, t))
    np.savetxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\data2.txt'%folname, data2, header = "theta 1     Omega 1     Theta 2     Omega 2       Time")
    np.savetxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\coordinates2.txt'%folname, coordt2, header = "a1       b1      a2      b2     Time")
    
    return (a1, b1, a2, b2)

if __name__ == '__main__':

    # Equations:
    # o1' = ( -g * (2*m1 + m2) * np.sin(th1) - m2 * g * np.sin(th1 - 2 * th2) - 2 * np.sin(th1 - th2) * m2 * (o2 ** 2 * l2 + o1 ** 2 * l1 * np.cos(th1 - th2)) ) / ( l1 * (2 * m1 + m2 - m2 * np.cos(2 * (th1 - th2))) )
    # o2' = ( 2 * np.sin(th1 - th2)*(o1 ** 2 * l1 * (m1 + m2) + (m1 + m2) * g * np.cos(th1) + o2 ** 2 * l2 * m2 * np.cos(th1 - th2)) ) / ( l2 * (2 * m1 + m2 - m2 * np.cos(2 * (th1 - th2))) )
    # th1' = o1
    # th2' = o2
    
    consts=[9.8, 2, 2, 1, 2]                          # Constants
    y0 = [np.pi*89/180, 0, np.pi*55/180, 0]         # Initial Value

    consts2=[9.8, 2, 2, 1, 2]
    y02 = [np.pi*78/180, 0, np.pi*55/180+0.0001, 0]

    t = np.linspace(0, 60, 6000)                      # t or x Range

    sol_AN = odeint(func, y0, t, args=(consts,))
    sol_RK = rk4(y0=y0, x=t, consts=consts)
    sol_RK2 = rk4(y0=y02, x=t, consts=consts2)

    folname = "({:.5}, {:.5}, {:.7})".format(str(sol_RK[0, 0]), str(sol_RK[0, 2]), str(float(y02[2]-y0[2])))
    os.makedirs(r'C:\Users\jains\Desktop\Double Pendulum\%s'%folname, exist_ok=True)
    
    # Trajectory
    x1, y1, x2, y2 = track(sol_RK, folname)

    # Additional Data: If Applicable
    a1, b1, a2, b2 = addtrack(sol_RK2, folname)

    chanim.run(folname, r=True, consts = (consts[1], consts[2]), diffmass = (consts2[1], consts2[2]))

    # data = np.column_stack((t, sol_RK[:, 0], sol_RK[:, 1], sol_RK[:, 2], sol_RK[:, 3], sol_AN[:, 0], sol_AN[:, 1], sol_AN[:, 2], sol_AN[:, 3]))
    # np.savetxt(r'C:\Users\jains\Desktop\Double Pendulum\odedata.txt', data, header = 'Time, first 4 col. - Data, last 4 col. - ODE')
    
    # plt.plot(t, sol_RK[:, 0], label = "RK4")
    # plt.plot(t, sol_AN[:, 0], label = "ODEINT", linestyle = '--')
    # plt.legend()
    # plt.show()

    plt.plot(sol_RK[:, 0], sol_RK[:, 2])
    plt.show()
    # https://www.reddit.com/r/askscience/comments/2smxyo/specifically_why_cant_partial_differential/