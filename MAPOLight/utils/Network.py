# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:21:57 2022

@author: Wang Chong
"""
import os
import sys
import sumolib
import numpy as np
from collections import defaultdict

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")

#LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ
import libsumo as traci

LIBSUMO = True
MIN_GAP = 7.5 #车辆最小间距

class Network:
    def __init__(self,netfile):
        '''
       ____|±11|___|±12|___
       _±1_ e4 _±3_ e8  _±5_
       ____|±9|____|±10|____
       _±2_ e3 _±4_ e7  _±6_
           |±7|    |±8 |
        '''
        #self.net = sumolib.net.readNet(currentPath+"\\sumo_files\\network.net.xml")
        if(len(netfile)==0):
            #current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            #netfile = current_dir+'/sumo_files/network.net.xml'
            raise Exception("network file not found!")
        #self.net = sumolib.net.readNet(netfile)
        
        self.tls_to_lane = dict()
        self.tls_size = dict()
        self.neignbors = dict()
        
        self.net = sumolib.net.readNet(netfile)
        
        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', netfile])  # Start only to retrieve traffic light information
            conn = traci
        
        self.ts_ids = list(conn.trafficlight.getIDList())
        
        print("ids:",self.ts_ids)
        
        for tid in self.ts_ids:
            self.tls_to_lane.update({tid:list(dict.fromkeys(conn.trafficlight.getControlledLanes(tid)))})
        
        print(self.tls_to_lane)

        self.net_coords = self.get_net_coords()
        
        #print("test!!!!!!!!!")
        nodes = self.net.getNodes()
        for node in nodes:
            if(node.getID() in self.ts_ids):
                #print("---",node.getID())
                neignbors = node.getNeighboringNodes()
                neignborset = set()
                for neignbor in neignbors:
                    if(neignbor.getID() in self.ts_ids):
                        neignborset.add(neignbor.getID())
                self.neignbors.update({node.getID():neignborset})
            #print("+++++++++++")
        #self.tls_size = self.get_size()
        
        print(self.neignbors)
        conn.close()

    def get_net_coords(self):
        net_coords = defaultdict(dict)
        
        
        for tls in self.tls_to_lane.keys():
            tls_encode = defaultdict(list)
            for lane_id in self.tls_to_lane[tls]:
                 
                begin,end = self.net.getLane(lane_id).getShape()
                lane_length = self.net.getLane(lane_id).getLength()
                cell_num = int(lane_length/MIN_GAP) #MIN_GAP=step
                
                #print(cell_num)
                #print(lane_length)
                lane_encode = [begin]
                for i in range(cell_num):
                    step_x = (end[0] - begin[0])/cell_num
                    step_y = (end[1] - begin[1])/cell_num
                    pos_x = begin[0] + step_x / 2 + i * step_x
                    pos_y = begin[1] + step_y / 2 + i * step_y
                    pos = (pos_x,pos_y)
                    lane_encode.append(pos)
                lane_encode.append(end)
                tls_encode[lane_id]=lane_encode
            net_coords[tls] = tls_encode

        return net_coords
        
    def build_maskarr(self,tls:str):
        lanes = dict()
        for lane in self.net_coords[tls].keys():
            length = len(self.net_coords[tls][lane])
            lanes.update({lane:np.zeros((length,))})
        return lanes