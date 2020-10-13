# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:07:09 2020

@author: Elias
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:50:25 2020

@author: Elias
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import csv
import argparse
import bct.utils as bctu
import bct.algorithms as bct
import scipy.stats as st
import ppscore as pps

class RunBCT:
    
    def __init__(self, corrmatrix, sessionfile , verbosity=1):

        # the npy file containing the raw 3d correlation files for each session
        # for the target ROI's
        self.corr_matrix_file = corrmatrix
    
        # file containing info on each session in the npy file
        # Participant,Timepoint,Group,Processing Deriv
        # header: 
        self.session_file = sessionfile
        
        # default is set to 1
        self.verbosity = verbosity
    
    def load_files(self):
        # load all the data files
        
        # read in the raw 3d correlation matrix, adjacency matrix for 
        # each session (3rd dimension)
        try:
            self.corr_matrix = np.load(self.corr_matrix_file)
        except FileNotFoundError as fnf_error:
            print(fnf_error)
            exit(1)
        
        try:
            self.session_info = pd.read_csv(self.session_file)
        except FileNotFoundError as fnf_error:
            print(fnf_error)
            exit(1)
        
        return True    
    
    def create_output_file(self, datafile):
        
        # create info df from the session file (id, grp, tp)     
        self.session_info = pd.read_csv(self.session_file)
        #self.session_info = self.session_info.drop(self.session_info.columns[[3,4]], axis=1)
        
        #join with data file
        self.output = self.session_info.join(datafile)
            
    def get_node_deg_cent(self, corrmatrix, norm):
        '''
        Degree Centrality: number links connected to each node/roi
        
        inputs
            corrmatrix: thresholded and binarized corr matrix [roi x roi x session]
            norm: set to 1 if looking to normalize the deg centrality measure, otherwise set to 0
                
        output: sum of links each node has in each session [session x roi deg]
            
        '''            
        self.roi_deg = bct.degrees_und(corrmatrix)
        
        if norm == 1:
            
            self.roi_deg = self.roi_deg/(corrmatrix.shape[0] - 1)
            
        #transpose to make it [session x roi]
        self.roi_deg = np.transpose(self.roi_deg)   
        #convert roi_str to pandas
        self.roi_deg = pd.DataFrame(self.roi_deg)

        self.create_output_file(self.roi_deg)
        
        return self.output

    def get_session_deg_cent(self, corrmatrix, norm):
        '''
        average numner of links per node in each session
        
        inputs
            corrmatrix: thresholded and binarized corr matrix [roi x roi x session]
            norm: set to 1 if looking to normalize the deg centrality measure, otherwise set to 0
                
        output: sum of links each node has in each session [session x roi deg]
            
        '''        
        self.roi_deg = bct.degrees_und(corrmatrix)
        
        if norm == 1:
            
            self.roi_deg = self.roi_deg/(corrmatrix.shape[0] - 1)
            
        #average across all roi's
        self.session_deg = np.average(self.roi_deg, axis = 0)
        #convert roi_str to pandas
        self.session_deg = pd.DataFrame(self.session_deg)

        self.create_output_file(self.session_deg)
        
        return self.output  
    
    def get_node_str(self, corrmatrix):

        '''
        Gets the average strength of of connectivity at each node
        
        INPUT: weighted networks
            
        '''        
        
        #STRENGTH
        ##gets the avg strength of connections for each roi
        #input: corr matrix [roi x roi x session] 
        
        #remove 0's
        corrmatrix[corrmatrix == 0] = np.nan
        
        #trasform to Z for averging       
        zmatrix = np.arctanh(corrmatrix)
        
        #output: the average str of connections PER ROI [roi str x session] 
        self.roi_str = np.nanmean(zmatrix, axis = 0)
        
        #back to r
        self.roi_str = np.tanh(self.roi_str)
        
        #transpose to make it [session x roi]
        self.roi_str = np.transpose(self.roi_str)
        #convert roi_str to pandas
        self.roi_str = pd.DataFrame(self.roi_str)

        self.create_output_file(self.roi_str)
        
        return self.output
    
    def get_session_str(self, corrmatrix):
        '''
        Gets the average strength of of connectivity across each scan
        
        INPUT: weighted networks
            
        '''         
        #remove 0's
        corrmatrix[corrmatrix == 0] = np.nan
        
        #trasform to Z for averging       
        zmatrix = np.arctanh(corrmatrix)
        
        #the average str of connections PER ROI [roi str x session] 
        self.roi_str = np.nanmean(zmatrix, axis = 0)
        
        #average across all roi/nodes to get average strength of the SESSION 
        self.session_str = np.average(self.roi_str, axis = 0)    
        
        #convert back to pearsons R
        self.session_str = np.tanh(self.session_str)
              
        #convert roi_str to pandas
        self.session_str = pd.DataFrame(self.session_str)
        
        self.create_output_file(self.session_str)
        
        return self.output
    
    def get_node_div(self, corrmatrix):
        '''
        Gets the average varaince at each node
        
        INPUT: weighted networks
            
        ''' 
        #diversity
        ##gets the var of connections for each roi 
        #input: corr matrix [roi x roi x session] 
        corrmatrix[corrmatrix == 0] = np.nan
        
        #trasform to Z for averging       
        zmatrix = np.arctanh(corrmatrix)
        
        #output: the average div of connections PER ROI [roi str x session] 
        self.roi_div = np.nanvar(corrmatrix, axis = 0)
        
        #convert back to pearsons R
        self.roi_div = np.tanh(self.roi_div)
        
        #transpose to make it [session x roi]
        self.roi_div = np.transpose(self.roi_div)
        #convert roi_str to pandas
        self.roi_div = pd.DataFrame(self.roi_div)
        
        self.create_output_file(self.roi_div)
        
        return self.output
      
    def get_session_div(self, corrmatrix):

        '''
        Gets the average variance at each session
        
        INPUT: weighted networks
            
        '''         
        #diversity
        ##gets the var of connections for each roi 
        #input: corr matrix [roi x roi x session] 
        
        #output: the average div of connections PER ROI [roi str x session]       
        corrmatrix[corrmatrix == 0] = np.nan
        
        #trasform to Z for averging       
        zmatrix = np.arctanh(corrmatrix)
               
        self.roi_div = np.nanvar(zmatrix, axis = 0)
        
        #output: average again to get average div of the SESSION
        self.session_div = np.average(self.roi_div, axis = 0)
        
        #convert back to pearsons R
        self.session_div = np.tanh(self.session_div)
        
        #convert to pandas
        self.session_div = pd.DataFrame(self.session_div)
        
        self.create_output_file(self.session_div)
        
        return self.output
        
    def get_node_entropy(self, corrmatrix):
        
        ###Wavelet entropy (univariate)
        
    
        #setup target array to hold entropy values
        self.roi_entropy = np.zeros((self.corr_matrix.shape[0], self.corr_matrix.shape[2]))
        
        #inputs
        # [roi x roi x session]
        # ci = community matrix from bct
        for session in range(self.corr_matrix.shape[2]):
            
            #get unsigned corr data for each session
            self.corr = self.corr_matrix[:,:,session]
            #use unsigned corr matrix to get community index
            self.ci = bct.community_louvain(self.corr_matrix[:,:,session])
            self.session_ci = self.ci[0]
            
            temp = bct.diversity_coef_sign(self.corr,self.session_ci) 
            temp = temp[0]
            
            for value in range(len(temp)):
                
                self.roi_entropy[value,session] = temp[value]
        
        #transpose to make it [session x roi]
        self.roi_entropy = np.transpose(self.roi_entropy)
        #convert to pandas
        self.roi_entropy = pd.DataFrame(self.roi_entropy)
        
        self.create_output_file(self.roi_entropy)
        
        
        #returns entropy for each roi per session and avg entropy for session
        return self.output
    
    def get_session_entropy(self, corrmatrix):
        
        ###Wavelet entropy for whole session(univariate)
    
        self.get_node_entropy(corrmatrix)
        
        self.session_entropy = np.average(self.roi_entropy, axis = 1)
        
        #convert to pandas
        self.session_entropy = pd.DataFrame(self.session_entropy)
        
        self.create_output_file(self.session_entropy)
        
        
        #returns entropy for each roi per session and avg entropy for session
        return self.output
    
    def get_components(self):
        '''
        Need a binarized undirected network as input
        returns a matrix of component sizes [roi x session]
            
        '''
        #setup array to hold component size info [roi x session]
        self.comp_matrix = np.zeros((self.corr_matrix.shape[0],self.corr_matrix.shape[2]))
        
        #run loop over thresholded and binarized matrix
        for session in range(self.corr_matrix.shape[2]):
            
            session_matrix = self.corr_matrix_thr_bin[:,:,session]
            session_comps = bct.get_components(session_matrix)
            comps = session_comps[0]
            
            self.comp_matrix[:,session] = comps
        
        return self.comp_matrix
    
    def apply_threshold_bin(self, thr, binarize):
        
        '''
        applies absolute threshold (thr) to corr matrix
        
        if binarize is == 1, binarizes matrix
        
        '''

        self.corr_matrix_thr = np.zeros((self.corr_matrix.shape[0],
                                         self.corr_matrix.shape[0],
                                         self.corr_matrix.shape[2]))
        
        for session in range(self.corr_matrix.shape[2]):
           
           session_matrix = self.corr_matrix[:,:,session]
           session_matrix_th = bctu.threshold_absolute(session_matrix,thr)
           
           self.corr_matrix_thr[:,:,session] = session_matrix_th 
        
        if binarize == 1:   
            return self.binzrize()
        else:          
            return self.corr_matrix_thr
    
    def binzrize(self):
        
        self.corr_matrix_thr_bin = np.zeros((self.corr_matrix_thr.shape[0],
                                             self.corr_matrix_thr.shape[0],
                                             self.corr_matrix_thr.shape[2]))
        
        for session in range(self.corr_matrix_thr.shape[2]):
           
           session_matrix = self.corr_matrix_thr[:,:,session]
           session_matrix_bin = bctu.binarize(session_matrix)
           
           self.corr_matrix_thr_bin[:,:,session] = session_matrix_bin 
        
        return self.corr_matrix_thr_bin
    
    def get_density(self, corrmatrix):
        
        '''
        computes graph density,the fraction of present connections to possible connections.
        
        INPUT: either binarized or weighted matrix
        OUTPUT: density
        '''
        
        #build container for cluter values
        self.density = np.zeros((corrmatrix.shape[2]))

        #run over each session
        for session in range(corrmatrix.shape[2]):
            
            session_data = corrmatrix[:,:,session]
            session_info = bct.density_und(session_data)
            
            den_val = session_info[0]
            
            #add to matrix
            self.density[session] = den_val
        
        #transpose and pandas 
        self.density = pd.DataFrame(self.density)
        #create output
        self.create_output_file(self.density)
        
        return self.output
    
    def get_node_clust_coef_bu(self, corrmatrix):
        '''
        computes clusterin coef for each node 
        
        The clustering coefficient is the fraction of triangles around a node
        (equiv. the fraction of nodes neighbors that are neighbors of each other).
    
        Parameters
        ----------
        input : NxN np.ndarray BINERIZED connection matrix
    
        Returns
        -------
        node_cluster_matrix = [session x roi]
        '''
        #build container for cluter values
        self.node_clust_matrix_bu = np.zeros((corrmatrix.shape[0],
                                         corrmatrix.shape[2]))

        #run over each session
        for session in range(corrmatrix.shape[2]):
            
            session_data = corrmatrix[:,:,session]
            session_clust = bct.clustering_coef_bu(session_data)
            
            #build clust matrix
            self.node_clust_matrix_bu[:,session] = session_clust
        
        #transpose and pandas
        self.node_clust_matrix_bu = np.transpose(self.node_clust_matrix_bu) 
        self.node_clust_matrix_bu = pd.DataFrame(self.node_clust_matrix_bu)
        #create output
        self.create_output_file(self.node_clust_matrix_bu)
        
        return self.output
        
    def get_session_clust_coef_bu(self, corrmatrix):    
        '''
        computes clusterin coef across all nodes for each session
        
    
        Parameters
        ----------
        input : NxN np.ndarray BINERIZED UWEIGHTED connection matrix
    
        Returns
        -------
        session clusters matrix = [session]
        '''
        #build container for cluter values
        self.node_clust_matrix_bu = np.zeros((corrmatrix.shape[0],
                                              corrmatrix.shape[2]))

        #run over each session
        for session in range(corrmatrix.shape[2]):
            
            session_data = corrmatrix[:,:,session]
            session_clust = bct.clustering_coef_bu(session_data)
            
            #build clust matrix
            self.node_clust_matrix_bu[:,session] = session_clust
        
        #average across all nodes in session
        self.session_clust_matrix_bu = np.average(self.node_clust_matrix_bu, axis = 0)
        self.session_clust_matrix_bu = pd.DataFrame(self.session_clust_matrix_bu)
        
        self.create_output_file(self.session_clust_matrix_bu)
        
        return self.output
        
    def get_community_louvain(self, corrmatrix):
        '''
        computes modularity statistic for each session
        
    
        Parameters
        ----------
        input : NxN np.ndarray BINERIZEDconnection matrix
    
        Returns
        -------
        session q-stat  = [session]
        '''       
        self.comm_matrix = np.zeros((corrmatrix.shape[0],corrmatrix.shape[2]))
        self.q_list = np.zeros((corrmatrix.shape[2]))
        
        #run loop over thresholded and binarized matrix
        for session in range(corrmatrix.shape[2]):
            
            session_data = corrmatrix[:,:,session]
            session_struct = bct.community_louvain(session_data, ci=None,B='modularity',seed=np.random.RandomState(1))
            ci = session_struct[0]
            q_stat = session_struct[1]
            
            self.comm_matrix[:,session] = ci
            self.q_list[session] = q_stat

            
        self.comm_matrix = np.transpose(self.comm_matrix)
        self.comm_matrix = pd.DataFrame(self.comm_matrix)
        self.q_list = pd.DataFrame(self.q_list)
        
        self.create_output_file(self.comm_matrix)
        
        self.comm_matrix = self.output
        
        self.create_output_file(self.q_list)
        
        self.q_list = self.output
        
        return self.comm_matrix, self.q_list
    
    def get_char_path_len(self, corrmatrix):
        '''
        computes char path length
        
    
        Parameters
        ----------
        input : NxN np.ndarray BINERIZED connection matrix
    
        Returns
        -------
        session clusters matrix = [session]
        '''
        self.dist_matrix = np.zeros((corrmatrix.shape[0],
                                    corrmatrix.shape[0],
                                    corrmatrix.shape[2]))
        
        self.char_path = np.zeros((corrmatrix.shape[2],2))
        
        #first make dist matrix
        for session in range(corrmatrix.shape[2]):
            
            session_data = corrmatrix[:,:,session]
            dist_matrix = bct.distance_bin(session_data)
            
            self.dist_matrix[:,:,session] = dist_matrix
            
        for session in range(corrmatrix.shape[2]):
            
            session_data = self.dist_matrix[:,:,session]
            char = bct.charpath(session_data, include_diagonal = False,
                                include_infinite= False)
            
            self.char_path[session, 0] = char[0]
            self.char_path[session, 1] = char[1]
    
        
        self.char_path = pd.DataFrame(self.char_path)
        
        self.create_output_file(self.char_path)
        
        
        return self.output
    
    def get_node_part_coef(self, corrmatrix):
        
        ###getting participant coeff
        # Participation coefficient is a measure of diversity of intermodular 
        # connections of individual nodes.
        
        #input: undirected network, community index vector
        #output :  list og par coeffs for each node in each session
        

        #setup target array to hold values
        self.roi_parc = np.zeros((self.corr_matrix.shape[0], self.corr_matrix.shape[2]))
        
        #inputs
        # [roi x roi x session]
        # ci = community matrix from bct
        for session in range(self.corr_matrix.shape[2]):
            
            #get unsigned corr data for each session
            self.corr = self.corr_matrix[:,:,session]
            #use unsigned corr matrix to get community index
            self.ci = bct.community_louvain(self.corr_matrix[:,:,session])
            self.session_ci = self.ci[0]
            
            temp = bct.participation_coef(self.corr,self.session_ci, degree = 'undirected') 

            
            for value in range(len(temp)):
                
                self.roi_parc[value,session] = temp[value]
        
        #transpose to make it [session x roi]
        self.roi_parc = np.transpose(self.roi_parc)
        #convert to pandas
        self.roi_parc = pd.DataFrame(self.roi_parc)
        
        self.create_output_file(self.roi_parc)
        
        
        #returns entropy for each roi per session and avg entropy for session
        return self.output
    
    def get_session_part_coef(self, corrmatrix):
        ###averaging across nodes to get per session part coeff value
    
        self.get_node_part_coef(corrmatrix)
        
        self.session_parc = np.average(self.roi_parc, axis = 1)
        
        #convert to pandas
        self.session_parc = pd.DataFrame(self.session_parc)
        
        self.create_output_file(self.session_parc)
        
        return self.output
    
    def get_node_bet_cent(self, corrmatrix, norm):
       
        '''
        
         Node betweenness centrality is the fraction of all shortest paths in 
         the network that contain a given node. Nodes with high values of 
         betweenness centrality participate in a large number of shortest paths.
         
         Input:    binary connection matrix.
         
         Output:   node betweenness centrality vector.
         
        '''
        #build container for session values
        self.node_bet_matrix = np.zeros((corrmatrix.shape[0],
                                          corrmatrix.shape[2]))
        
         #run over each session
        for session in range(corrmatrix.shape[2]):
             
             session_data = corrmatrix[:,:,session]
             session_bet = bct.betweenness_bin(session_data)
             
             if norm == 1:
                 
                 session_bet = session_bet/((corrmatrix.shape[0] - 1)*(corrmatrix.shape[0] - 2)/4)
             
             #build bet matrix
             self.node_bet_matrix[:,session] = session_bet
         
        #transpose and pandas
        self.node_bet_matrix = np.transpose(self.node_bet_matrix) 
        self.node_bet_matrix = pd.DataFrame(self.node_bet_matrix)
        
        #create output
        self.create_output_file(self.node_bet_matrix)
         
        return self.output
    
    def get_session_bet_cent(self, corrmatrix, norm):
         '''
        
         Node betweenness centrality is the fraction of all shortest paths in 
         the network that contain a given node. Nodes with high values of 
         betweenness centrality participate in a large number of shortest paths.
         
         Input:    binary (directed/undirected) connection matrix.
         
         Output:   node betweenness centrality vector.
         
        '''
        #build container for session values
         self.node_bet_matrix = np.zeros((corrmatrix.shape[0],
                                          corrmatrix.shape[2]))
        
         #run over each session
         for session in range(corrmatrix.shape[2]):
             
             session_data = corrmatrix[:,:,session]
             session_bet = bct.betweenness_bin(session_data)
             
             if norm == 1:
                 
                 session_bet = session_bet/((corrmatrix.shape[0] - 1)*(corrmatrix.shape[0] - 2)/4)
             
             #build clust matrix
             self.node_bet_matrix[:,session] = session_bet
         
         #average across all nodes in session
         self.session_bet_matrix = np.average(self.node_bet_matrix, axis = 0)
         self.session_bet_matrix = pd.DataFrame(self.session_bet_matrix)
         
         self.create_output_file(self.session_bet_matrix)
         
         return self.output
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        