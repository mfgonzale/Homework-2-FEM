from numpy import array, sqrt
from sympy.matrices import Matrix

def cst_planestress(xy, properties):
    E = properties["E"]
    nu = properties["nu"]
    bx = properties["bx"]
    by = properties["by"]
    t = properties["t"]

    E_sigma = E / (1-nu**2) * array([
        [1, nu, 0],
        [nu, 1, 0],
        [0,0,(1-nu)/2]
        ])

    x0 = xy[0,0]
    x1 = xy[1,0]
    x2 = xy[2,0]
    y0 = xy[0,1]
    y1 = xy[1,1]
    y2 = xy[2,1]

    l0 = sqrt((x1-x2)**2 + (y1-y2)**2)
    l1 = sqrt((x0-x2)**2 + (y0-y2)**2)
    l2 = sqrt((x1-x0)**2 + (y1-y0)**2)

    # Formula de Hero of Alexandria

    s = (l0 + l1 + l2)/2
    Ae = sqrt((s-l0)*(s-l1)*(s-l2)*s)
    
    dz0_dx = (y1 - y2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dz0_dy = (-x1 + x2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dz1_dx = (-y0 + y2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dz1_dy = (x0 - x2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dz2_dx = (y0 - y1)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dz2_dy = (-x0 + x1)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)

    B = array([
    [dz0_dx, 0, dz1_dx, 0, dz2_dx, 0],
    [0, dz0_dy, 0, dz1_dy, 0, dz2_dy],
    [dz0_dy, dz0_dx, dz1_dy, dz1_dx, dz2_dy, dz2_dx]
    ])

    ke = B.T @ E_sigma @ B * Ae * t
    fe = (Ae*t/3) * array([bx, by ,bx, by, bx, by])
    
    return ke, fe


def cst_planestress_post(xy, u_e, properties):
    
    E = properties["E"]
    nu = properties["nu"]
    bx = properties["bx"]
    by = properties["by"]
    t = properties["t"]
    
    E_sigma = E / (1-nu**2) * array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    
    x0 = xy[0,0]
    x1 = xy[1,0]
    x2 = xy[2,0]
    y0 = xy[0,1]
    y1 = xy[1,1]
    y2 = xy[2,1]
    
    l0 = sqrt((x1-x2)**2 + (y1-y2)**2)
    l1 = sqrt((x0-x2)**2 + (y0-y2)**2)
    l2 = sqrt((x1-x0)**2 + (y1-y0)**2)
    
    #Formula de Heron de Alejandria
    
    s = (l0+l1+l2)/2
    
    Ae = sqrt( (s-l0)*(s-l1)*(s-l2)*s )
    
    dzeta0_dx = (y1 - y2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dzeta0_dy = -(x1 - x2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dzeta1_dx = -(y0 - y2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dzeta1_dy = (x0 - x2)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dzeta2_dx = (y0 - y1)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    dzeta2_dy = -(x0 - x1)/(x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)
    
    B = array([[dzeta0_dx,0,dzeta1_dx,0,dzeta2_dx,0],[0,dzeta0_dy,0,dzeta1_dy,0,dzeta2_dy],[dzeta0_dy,dzeta0_dx,dzeta1_dy,dzeta1_dx,dzeta2_dy,dzeta2_dx]])
   
    epsilon = B @ u_e
    sigma = E_sigma @ epsilon
    return epsilon, sigma


"""
xy = array([
    [0,0],
    [1,0],
    [0,1]
    ])

properties = {}
properties["E"] = 1
properties["nu"] = 0.25
properties["bx"] = 0
properties["by"] = 1
properties["t"] = 1

ke, fe = cst_planestress(xy, properties)

print(f"ke = {ke}")
print(f"fe = {fe}")
"""
