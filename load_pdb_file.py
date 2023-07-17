import numpy as np
import os
from CelluloseSim_pdb import *

path = os.path.join('/home/esrf/jiliu/Desktop/code_v5/test_analysis_jupyter/Aalto_ana/model_fitting',
                    'cellulose_pdb')
cellulose_Ia_fn = os.path.join(path,'ia.out.pdb')
cellulose_Ib_fn = os.path.join(path,'ib.out.pdb')
cellulose_II_fn = os.path.join(path,'i2.out.pdb')
cellulose_III_fn = os.path.join(path,'i3.out.pdb')

def load_fibril_pdb_file(fn,max_rows=84):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=max_rows,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=max_rows,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label
    cellulose_param['x'] = atom_coord[:,0]
    cellulose_param['y'] = atom_coord[:,1]
    cellulose_param['z'] = atom_coord[:,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def ca_load_pdb_file(fn):
    #has problem, xyz order is not correct
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=42,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=42,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = b
    cellulose_param['b'] = c
    cellulose_param['c'] = a
    cellulose_param['alpha'] = beta
    cellulose_param['beta']  = gamma
    cellulose_param['gamma'] = alpha
    cellulose_param['atom_symbol'] = atom_label
    cellulose_param['x'] = atom_coord[:,1]
    cellulose_param['y'] = atom_coord[:,2]
    cellulose_param['z'] = atom_coord[:,0]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    atom_coord1 = np.roll(atom_coord,1,axis=0)
    fract_mat = np.matmul(atom_coord1,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def cb_load_pdb_file(fn):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label
    cellulose_param['x'] = atom_coord[:,0]
    cellulose_param['y'] = atom_coord[:,1]
    cellulose_param['z'] = atom_coord[:,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def c2_load_pdb_file(fn):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label
    cellulose_param['x'] = atom_coord[:,0]
    cellulose_param['y'] = atom_coord[:,1]
    cellulose_param['z'] = atom_coord[:,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def c2_chain1_load_pdb_file(fn):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label[::2]
    cellulose_param['x'] = atom_coord[::2,0]
    cellulose_param['y'] = atom_coord[::2,1]
    cellulose_param['z'] = atom_coord[::2,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def c2_chain2_load_pdb_file(fn):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=84,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label[1::2]
    cellulose_param['x'] = atom_coord[1::2,0]
    cellulose_param['y'] = atom_coord[1::2,1]
    cellulose_param['z'] = atom_coord[1::2,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param

def c3_load_pdb_file(fn):
    (a,b,c,alpha,beta,gamma) = np.loadtxt(fn,skiprows=1,max_rows=1,usecols=(1,2,3,4,5,6))
    atom_label = np.loadtxt(fn,skiprows=2,max_rows=42,usecols=(2),dtype=str)
    atom_coord = np.loadtxt(fn,skiprows=2,max_rows=42,usecols=(5,6,7))
    cellulose_param = {}
    cellulose_param['a'] = a
    cellulose_param['b'] = b
    cellulose_param['c'] = c
    cellulose_param['alpha'] = alpha
    cellulose_param['beta']  = beta
    cellulose_param['gamma'] = gamma
    cellulose_param['atom_symbol'] = atom_label
    cellulose_param['x'] = atom_coord[:,0]
    cellulose_param['y'] = atom_coord[:,1]
    cellulose_param['z'] = atom_coord[:,2]
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    rmat = np.vstack((rvec['a'],rvec['b'],rvec['c']))
    fract_mat = np.matmul(atom_coord,np.linalg.inv(rmat))
    cellulose_param['fract_x'] = fract_mat[:,0]
    cellulose_param['fract_y'] = fract_mat[:,1]
    cellulose_param['fract_z'] = fract_mat[:,2]
    cellulose_param['rvec'] = rvec
    cellulose_param['qvec'] = qvec
    cellulose_param['symmetry_coefficient1'] = [[1,1,1]]
    cellulose_param['symmetry_coefficient2'] = [[0,0,0]]
    return cellulose_param