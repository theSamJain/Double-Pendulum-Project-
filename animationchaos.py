import numpy as np
import pygame 
from pygame.locals import *
import sys

def run(folname, consts, r=True, diffmass = None):
    
    def rend(x1, y1, x2, y2, consts, s):
        scale1 = 100
        x1 = scale1 * x1 + off[0]
        y1 = scale1 * y1 + off[1]
        x2 = scale1 * x2 + off[0]
        y2 = scale1 * y2 + off[1]
        
        point1 = (x1, y1)
        point2 = (x2, y2)

        m1, m2 = consts
        scale2 = 10

        if (s == 1):
            if prevp:
                xp, yp = prevp
                pygame.draw.line(trace, lbl, (xp, yp), (x2, y2), 2)

            screen.blit(trace, (0, 0))
            pygame.draw.line(screen, w, off, point1, 3)
            pygame.draw.line(screen, w, point1, point2, 3)
            pygame.draw.circle(screen, w, off , int(scale2/2))
            pygame.draw.circle(screen, r, point1 , int(m1 * scale2))
            pygame.draw.circle(screen, lbl, point2 , int(m2 * scale2))
        
        elif (s == 2):
            if prevp2:
                xp, yp = prevp2
                pygame.draw.line(trace, p, (xp, yp), (x2, y2), 2)
            
            if diffmass:
                m1, m2 = diffmass
            pygame.draw.line(screen, w, off, point1, 3)
            pygame.draw.line(screen, w, point1, point2, 3)
            pygame.draw.circle(screen, w, off , int(scale2/2))
            pygame.draw.circle(screen, bl, point1 , int(m1 * scale2))
            pygame.draw.circle(screen, p, point2 , int(m2 * scale2))

        return (x2, y2)

    w = (248, 248, 255)     # Off White
    b = (50, 50, 50)        # Gray
    teal = (0, 200, 200)    # Time Teal
    r = (255, 165, 0)       # Bob Orange
    bl = (210, 180, 140)    # Bob Tan
    lbl = (100, 200, 255)   # Trail light blue
    p = (200, 100, 231)     # Trail Purple

    x1, y1, x2, y2, t = np.loadtxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\coordinates.txt'%folname, unpack = True)
    xx1, yy1, xx2, yy2, t = np.loadtxt(r'C:\Users\jains\Desktop\Double Pendulum\%s\coordinates2.txt'%folname, unpack = True)    
    wid, hyt = 1000, 750
    off = (500, 300)
    screen = pygame.display.set_mode((wid,hyt))
    screen.fill(b)

    prevp = None
    prevp2 = None
    trace = screen.copy()
    pygame.display.update()
    clock = pygame.time.Clock()
    pygame.font.init()
    ft = pygame.font.SysFont("Showcard Gothic", 40)
    for i in range(len(x1)):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        prevp = rend(x1[i], -y1[i], x2[i], -y2[i], consts, s = 1)
        prevp2 = rend(xx1[i], -yy1[i], xx2[i], -yy2[i], consts, s = 2)

        time = "Time: {:} seconds".format(round(t[i],1))
        tstamp = ft.render(time, False, teal)
        screen.blit(tstamp, (50,50))

        clock.tick(200)
        pygame.display.update()