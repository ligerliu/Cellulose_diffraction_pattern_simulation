import numpy as np
from scipy.special import j0,j1

def rotate_matrix(alpha,beta,gamma):
    '''
    rotate matrix for vector in euclid space
    defined by three anlges rotating about different axis in degrees
    alpha rotate about x-aixs counter-clock wise in y-z plane
    beta rotate about y-axis counter-clock wise in x-z plane
    gamma rotate about z-axis counter-clock wise in x-y plane
    '''
    alpha = np.radians(alpha)
    beta  = np.radians(beta)
    gamma = np.radians(gamma)
    
    Rx = np.array([(1, 0, 0,),
                   (0, np.cos(alpha), -np.sin(alpha)),
                   (0, np.sin(alpha),  np.cos(alpha))])
    Ry = np.array([( np.cos(beta), 0, np.sin(beta)),
                   (0, 1, 0),
                   (-np.sin(beta), 0, np.cos(beta))])
    Rz = np.array([(np.cos(gamma), -np.sin(gamma), 0),
                   (np.sin(gamma),  np.cos(gamma), 0),
                   (0, 0 ,1)])
    return np.matmul(np.matmul(Rx,Ry),Rz)

def shift_phase(qvec,rvec):
    '''
    the displacement of particle in real space will lead to phase difference in fourier space
    rvec is displacement of real space vector
    qvec is displacement of fourier space vector
    '''
    return np.exp(1j*np.dot(qvec,rvec))

def Atomic_form_factor(q,
                        a1,a2,a3,a4,
                        b1,b2,b3,b4,
                        c):
    f = a1*np.exp(-b1*(q/4/np.pi)**2) + \
        a2*np.exp(-b2*(q/4/np.pi)**2) + \
        a3*np.exp(-b3*(q/4/np.pi)**2) + \
        a4*np.exp(-b4*(q/4/np.pi)**2) + c
    return f

def vec_cal(a,b,c,alpha,beta,gamma,ang_unit='degree'):
    '''
    calculate the real space vector, and reciprocal space vector 
    here for convenience of fiber diffraction calculation, we assign c vector
    (fiber axis) same to z-axis
    '''
    if ang_unit=='degree':
        alpha = np.radians(alpha)
        beta  = np.radians(beta)
        gamma = np.radians(gamma)
    else:
        pass
    a1 = a*np.sin(beta)
    a2 = 0.
    a3 = a*np.cos(beta)
    b3 = b*np.cos(alpha)
    b1 = (a*b*np.cos(gamma)-a3*b3)/a1
    b2 = np.sqrt(b**2-b1**2-b3**2)
    
    a_vec = np.array([a1, 0,a3])
    b_vec = np.array([b1,b2,b3])
    c_vec = np.array([ 0, 0, c])
    rvec = {'a' : a_vec,
            'b' : b_vec,
            'c' : c_vec}
    # s = 2*np.pi/V
    s = 2*np.pi/(c*(a1*b2-a2*b1))
    qa_vec = s*np.cross(b_vec,c_vec)
    qb_vec = s*np.cross(c_vec,a_vec)
    qc_vec = s*np.cross(a_vec,b_vec)
    qvec = {'A' : qa_vec,
            'B' : qb_vec,
            'C' : qc_vec}
    return rvec,qvec

def q_hkl(h,k,l,qvec):
    '''
    calculate the qvec of miller index
    '''
    qa_vec = qvec['A']
    qb_vec = qvec['B']
    qc_vec = qvec['C']
    q_vec = h*qa_vec+k*qb_vec+l*qc_vec
    q_hkl = np.linalg.norm(q_vec)
    return q_vec,q_hkl

def cellulose_atomic_factor(obj, q):
    '''
    calulcate scattering of one unit cell
    q,r include three base vector of unit cell in real space and reciprocal space
    obj_pos is object shift position to origin of unit cell
    abs_pos indicate the obj position is relative ratio of unit cell base vector or absolute real space position
    '''
    if obj == 'O':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 
                                                0.867, 32.9089, 0.2508])
    if obj == 'C':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 
                                                0.865, 51.6512, 0.2156])
    if obj == 'H':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 
                                                0.049879, 2.20159, 0.001305])
    f = Atomic_form_factor(q,a1,a2,a3,a4,b1,b2,b3,b4,c)
    return f

def cellulose_form_factor(obj, q, rvec, obj_pos, abs_pos = False):
    '''
    calulcate scattering of one unit cell
    q,r include three base vector of unit cell in real space and reciprocal space
    obj_pos is object shift position to origin of unit cell
    abs_pos indicate the obj position is relative ratio of unit cell base vector or absolute real space position
    '''
    if obj == 'O':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 
                                                0.867, 32.9089, 0.2508])
    if obj == 'C':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 
                                                0.865, 51.6512, 0.2156])
    if obj == 'H':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 
                                                0.049879, 2.20159, 0.001305])
    f = Atomic_form_factor(np.linalg.norm(q),a1,a2,a3,a4,b1,b2,b3,b4,c)
    
    if abs_pos == True:
        phase = np.exp(1j*np.dot(q,obj_pos))
    elif abs_pos == False:
        r = obj_pos[0]*rvec['a']+obj_pos[1]*rvec['b']+obj_pos[2]*rvec['c']
        phase = np.exp(1j*np.dot(q,r))
    return f*phase

def cellulose_hkl_coeff_pdb(rvec,qvec,h,k,l,pars):
    '''
    
    '''
    fhkl = 0+0*1j
    qhkl_vec,qhkl = q_hkl(h,k,l,qvec)
    for _ in range(len(pars['atom_symbol'])):
        obj_pos = np.array([pars['fract_x'][_],
                            pars['fract_y'][_],
                            pars['fract_z'][_]
                           ])
        fhkl += cellulose_form_factor(pars['atom_symbol'][_],qhkl_vec,rvec,obj_pos)
    return fhkl,qhkl

def peak_shape(q, p, mu, sigma):
    L = 2/np.pi/sigma/(1+(q-mu)**2/sigma**2)
    G = (4*np.log(2))**.5/sigma/np.pi**.5*np.exp(-4*np.log(2)*(q-i)**2/sigma**2)
    P = p*L+(1-p)*G
    return P/np.max(P)

def gaussian(q,mu,sigma):
    return np.exp(-(q-mu)**2/2/sigma**2)

def lorentz(q,mu,sigma):
    return 1/(1+(q-mu)**2/sigma**2)

def cellulose_structure_factor_pdb(q,h,k,l,pars,sigma):
    '''
    here calculate the powder diffraction pattern of cellulose
    which is spherical averaged intensity, should include lorentz correction with q**2
    '''
    
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = pars['alpha']
    beta  = pars['beta']
    gamma = pars['gamma']
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma,ang_unit='degree')
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(-l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(q)))
    for _ in range(hkl.shape[1]):
        fhkl,qhkl = cellulose_hkl_coeff_pdb(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars)
        if qhkl > np.max(q):
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(q,qhkl,sigma)
    return S/q**2

def cellulose_layer_line_pdb(qr,h,k,l,pars,sigma):
    '''
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma,ang_unit='radians')
    # only calculate the h, k index for l layer
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(qr)))
    for _ in range(hkl.shape[1]):
        #print(hkl[0,_],hkl[1,_],hkl[2,_])
        fhkl,qhkl = cellulose_hkl_coeff_pdb(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars)
        Qr_vec = hkl[0,_]*qvec['A']+hkl[1,_]*qvec['B']+hkl[2,_]*qvec['C']
        Qr = np.sqrt(Qr_vec[0]**2+Qr_vec[1]**2)
        if Qr > np.max(qr):
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(qr,Qr,sigma)
    return S/qr


def regenerate_cellulose_form_factor_2D_pdb(qx,qy,qz,pars,displacement,rota,rotb,rotc):
    '''
    for Ibeta cif sym_pos is [[1,1,1][-1,-1,1+.5]]
    for Ialpha cif sym_pos if [[1,1,1]]
    displacment is vector, here used to move mass center to origin.
    rota is angle rotate about x-axis, rotb is about yaxis, rotc is z-axis
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    rot = rotate_matrix(rota,rotb,rotc)
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma,ang_unit='radians')
    fhkl = qx.flatten()*0 + 1j*0
    for __ in range(len(qx.flatten())):
        q_vec = np.array([qx.flatten()[__],qy.flatten()[__],qz.flatten()[__]])        
        for _ in range(len(pars['atom_symbol'])):
            obj_pos = np.array([pars['fract_x'][_],
                                pars['fract_y'][_],
                                pars['fract_z'][_]
                               ])
            v = rvec['a']*obj_pos[0] +rvec['b']*obj_pos[1]+rvec['c']*obj_pos[2]
            v -= displacement
            v = np.matmul(rot,v)
            fhkl[__] += cellulose_form_factor(pars['atom_symbol'][_],q_vec,rvec,
                                              v,abs_pos=True)
            #here use abusolte position instead of fraction coordinate
    return fhkl

def regenerate_cellulose_form_factor_pdb(qx,qy,qz,pars,displacement,rota,rotb,rotc):
    '''
    for Ibeta cif sym_pos is [[1,1,1][-1,-1,1+.5]]
    for Ialpha cif sym_pos if [[1,1,1]]
    displacment is vector, here used to move mass center to origin.
    rota is angle rotate about x-axis, rotb is about yaxis, rotc is z-axis
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    rot = rotate_matrix(rota,rotb,rotc)
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma,ang_unit='radians')
    #fhkl = qx.flatten()*0 + 1j*0
    q_vec = np.vstack((qx.flatten(),qy.flatten(),qz.flatten())).T
    q = np.linalg.norm(q_vec,axis=1)
    j = 0
    for _ in range(len(pars['atom_symbol'])):
            obj_pos = np.array([pars['fract_x'][_],
                                pars['fract_y'][_],
                                pars['fract_z'][_]
                               ])
            v = rvec['a']*obj_pos[0] +rvec['b']*obj_pos[1]+rvec['c']*obj_pos[2]
            v -= displacement
            v = np.matmul(rot,v)
            v = v.reshape(3,1)
            f = cellulose_atomic_factor(pars['atom_symbol'][_],q)
            if j == 0:
                fhkl = f
                vhkl = v
            else:
                fhkl = np.vstack((fhkl,f))
                vhkl = np.hstack((vhkl,v))
            j += 1
            #here use abusolte position instead of fraction coordinate
    phase = np.exp(1j*np.matmul(q_vec,vhkl))
    fhkl = np.multiply(fhkl,phase.T)
    fhkl = np.sum(fhkl,axis=0)
    return fhkl.reshape(qx.shape)

def single_atom_scattering_factor(_,fract_x,fract_y,fract_z,rvec_a,rvec_b,rvec_c,atom_symbol,
                                  q,rot,displacement,q_vec):
    obj_pos = np.array([fract_x[_],
                        fract_y[_],
                        fract_z[_]
                           ])
    v = rvec_a*obj_pos[0] +rvec_b*obj_pos[1]+rvec_c*obj_pos[2]
    v -= displacement
    v = np.matmul(rot,v)
    v = v.reshape(3,1)
    f = cellulose_atomic_factor(atom_symbol[_],q)
    fhkl = f
    vhkl = v
    phase = np.exp(1j*np.matmul(q_vec,vhkl))
    fhkl = np.multiply(fhkl,phase.T)
    return fhkl
    
def parallel_regenerate_cellulose_form_factor_pdb(num_cores,qx,qy,qz,pars,displacement,rota,rotb,rotc):
    '''
    for Ibeta cif sym_pos is [[1,1,1][-1,-1,1+.5]]
    for Ialpha cif sym_pos if [[1,1,1]]
    displacment is vector, here used to move mass center to origin.
    rota is angle rotate about x-axis, rotb is about yaxis, rotc is z-axis
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    rot = rotate_matrix(rota,rotb,rotc)
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma,ang_unit='radians')
    #fhkl = qx.flatten()*0 + 1j*0
    q_vec = np.vstack((qx.flatten(),qy.flatten(),qz.flatten())).T
    q = np.linalg.norm(q_vec,axis=1)
    fract_x=pars['fract_x']
    fract_y= pars['fract_y']
    fract_z=pars['fract_z']
    rvec_a=rvec['a']
    rvec_b=rvec['b']
    rvec_c=rvec['c']
    atom_symbol=pars['atom_symbol']
    from multiprocessing import Pool
    from functools import partial
    partial_func = partial(single_atom_scattering_factor,fract_x=fract_x,
    fract_y=fract_y,
    fract_z=fract_z,
    rvec_a=rvec_a,
    rvec_b=rvec_b,
    rvec_c=rvec_c,
    atom_symbol=atom_symbol,q=q,rot=rot,displacement=displacement,q_vec=q_vec)
    p = Pool(num_cores)
    with p:
        res = p.map(partial_func,range(len(pars['atom_symbol'])))
    fhkl = np.array(res)
    fhkl = np.sum(fhkl,axis=0)
    return fhkl.reshape(qx.shape)
