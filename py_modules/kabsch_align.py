import math
import numpy
import numpy.linalg
import numpy as np

import logging

log = logging.getLogger('main.IterativeMeansAlign');
log.setLevel(logging.DEBUG);

class KabschAlign(object):

    def __init__(self):
        """
        Constructor
        """

    def kabsch(self, toXYZ, fromXYZ):
        """
        Input is a 3 x N array of coordinates.
        """
    # This file has been edited to produce identical results as the original matlab implementation.
        len1 = numpy.shape(fromXYZ);
        len2 = numpy.shape(toXYZ);

        if not(len1[1] == len2[1]):
            print('KABSCH: unequal array sizes');
            return;

        m1 = numpy.mean(fromXYZ, 1).reshape((len1[0],1)); # print numpy.shape(m1);
        m2 = numpy.mean(toXYZ, 1).reshape((len2[0],1)); 
        tmp1 = numpy.tile(m1,len1[1]);
        tmp2 = numpy.tile(m1,len2[1]);

        assert numpy.allclose(tmp1, tmp2);
        assert tmp1.shape == fromXYZ.shape;
        assert tmp2.shape == toXYZ.shape;
        t1 = fromXYZ - tmp1;
        t2 = toXYZ - tmp2;

        [u, s, wh] = numpy.linalg.svd(numpy.dot(t2,t1.T));
        w = wh.T;

        R = numpy.dot(numpy.dot(u,[[1, 0, 0],[0, 1, 0],[0, 0, numpy.linalg.det(numpy.dot(u,w.T))]]), w.T); 
        T = m2 - numpy.dot(R,m1);

        tmp3 = numpy.reshape(numpy.tile(T,(len2[1])),(len1[0],len1[1]));
        err = toXYZ - numpy.dot(R,fromXYZ) - tmp3; 

        #eRMSD = math.sqrt(sum(sum((numpy.dot(err,err.T))))/len2[1]); 
        eRMSD = math.sqrt(sum(sum(err**2))/len2[1]); 
        return (R, T, eRMSD, err.T);

    def wKabschDriver(self, toXYZ, fromXYZ, sMed=1.5, maxIter=20):
        scaleMed = sMed;
        weights = numpy.ones( numpy.shape(toXYZ)[1] ); #print 'weights: ', numpy.shape(weights);
        flagOut = 0;
        Rc = []; Tc = []; sigc = [];
        for itr in range(0, maxIter):
            [R, T, eRMSD, err] = self.wKabsch(toXYZ, fromXYZ, weights);
        Rc.append(R);
        Tc.append(T);
        tmp1 = numpy.reshape(numpy.tile(T, (numpy.shape(toXYZ[1]))), (numpy.shape(toXYZ)[0],numpy.shape(toXYZ)[1]));
        deltaR = numpy.array( numpy.dot(R, fromXYZ) + tmp1 - toXYZ ); #print 'deltaR shape: ', numpy.shape(deltaR);
        #print deltaR;
        #numpy.save('deltaR.npy', deltaR);
        nDeltaR = numpy.sqrt(numpy.sum(deltaR**2, axis = 0)); #print 'nDeltaR shape:', numpy.shape(nDeltaR);
        sig = scaleMed*numpy.median(nDeltaR);
        sigc.append(sig);
        weights = (sig**2)/((sig**2 + nDeltaR**2)**2); #print numpy.shape(weights);
        return ( R, T, eRMSD, err);
			
    def wKabsch(self, toXYZ, fromXYZ, weights):
        len1 = numpy.shape(fromXYZ); #print 'len1: ', len1;
        len2 = numpy.shape(toXYZ); #print 'len2: ', len2;

        if not(len1[1] == len2[1]):
            print('wKABSCH: unequal array sizes');
            return;

        dw = numpy.tile(weights, (3,1)); #print 'dw shape:', numpy.shape(dw);
        wFromXYZ = dw * fromXYZ; #print 'wFromXYZ shape: ', numpy.shape(wFromXYZ);
        wToXYZ = dw * toXYZ; # print 'wToXYZ shape: ', numpy.shape(wToXYZ);

        m1 = numpy.sum(wFromXYZ, 1) / numpy.sum(weights); #print numpy.shape(m1);
        m2 = numpy.sum(wToXYZ, 1) / numpy.sum(weights); #print numpy.shape(m2);

        tmp1 = numpy.reshape(numpy.tile(m1,(len1[1])), (len1[0],len1[1]));
        tmp2 = numpy.reshape(numpy.tile(m2,(len2[1])), (len2[0],len2[1])); 
        t1 = numpy.reshape(fromXYZ - tmp1, (len1[0], len1[1])); #print 't1 shape: ', numpy.shape(t1);
        t2 = numpy.reshape(toXYZ - tmp2, (len2[0],len2[1]));

        aa = numpy.zeros((3,3));
        for i in range(0, numpy.shape(t1)[1]):
            tmp = numpy.outer(t2[:,i],t1[:,i]); #print 'tmp shape: ', numpy.shape(tmp);
            aa = aa + numpy.multiply(weights[i], tmp);
            aa = aa/numpy.sum(weights);

        [u,s,wh] = numpy.linalg.svd(aa);
        w = wh.T;

        R = numpy.dot(numpy.dot(u,[[1, 0, 0],[0, 1, 0],[0, 0, numpy.linalg.det(numpy.dot(u,w.T))]]), w.T); 
        T = m2 - numpy.dot(R,m1); 

        tmp3 = numpy.reshape(numpy.tile(T,(len2[1])),(len1[0],len1[1]));
        err = toXYZ - numpy.dot(R,fromXYZ) - tmp3; 
        #eRMSD = math.sqrt(sum(sum((numpy.dot(err,err.T))))/len2[1]); 
        eRMSD = math.sqrt(sum(sum(err**2))/len2[1]); 
        return (R, T, eRMSD, err.T);


class IterativeMeansAlign(object):
	
	def __init__(self):
		"""
		Constructor
		"""

	def iterativeMeans(self, coords, eps, maxIter, mapped=False, fname='na',shape=[0,0,0]):

		if mapped:
			coords = np.memmap(fname, dtype='float64', mode='r+').reshape(shape) 
		# all coordinates are expected to be passed as a (Ns x 3 x Na)  array
		# where Na = number of atoms; Ns = number of snapshots
	
		# This file has been edited to produce identical results as the original matlab implementation.

		Ns = numpy.shape(coords)[0];
		dim = numpy.shape(coords)[1];
		Na = numpy.shape(coords)[2];
		
		log.debug('Shape of array in IterativeMeans: {0}'.format(numpy.shape(coords)));
		
		avgCoords = [];			# track average coordinates
		kalign = KabschAlign();		# initialize for use

		ok = 0;				# tracking convergence of iterative means
		itr = 1; 			# iteration number
		
		eRMSD = [];
		"""
		fig = plt.figure();
		ax = fig.gca(projection='3d');
		plt.ion();"""
		while not(ok):
			tmpRMSD = [];
			mnC = numpy.mean(coords, 0); 
			avgCoords.append(mnC);
			for i in range(0,Ns):
				fromXYZ = coords[i];
				[R, T, xRMSD, err] = kalign.kabsch(mnC, fromXYZ);
				tmpRMSD.append(xRMSD); 
				tmp = numpy.tile(T.flatten(), (Na,1)).T;
				pxyz = numpy.dot(R,fromXYZ) + tmp;  
				coords[i,:,:] = pxyz;
			eRMSD.append(numpy.array(tmpRMSD).T);
			newMnC = numpy.mean(coords,0); 
			err = math.sqrt(sum( (mnC.flatten()-newMnC.flatten())**2) )
			log.info('Iteration #{0} with an error of {1}'.format(itr, err))
			if err <= eps or itr == maxIter:
				ok = 1;
			itr = itr + 1;
		if mapped:
			del coords;
			coords = 0;
		return [itr,avgCoords,eRMSD,coords];