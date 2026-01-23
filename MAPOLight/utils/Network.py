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
#CELL_NUM = 60
MIN_GAP = 7.5 #车辆最小间距
#CELL_LENGTH = 5     # TODO： 统一网格数目或者网格长度
#LANE_LENGTH = 300
#MAX_SPEED = 50

class Network:
    def __init__(self,netfile):
        '''
       ____|±11|___|±12|___
       _±1_ e4 _±3_ e8  _±5_
       ____|±9|____|±10|____
       _±2_ e3 _±4_ e7  _±6_
           |±7|    |±8 |
          竖线：上左下右 
        '''
        #self.net = sumolib.net.readNet(currentPath+"\\sumo_files\\network.net.xml")
        if(len(netfile)==0):
            #current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #爷爷目录
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
        #print(self.net_coords[0])


        #print(self.size) (4,3,60,2)
        # 3 不如改成waiting time, velocity
        
        #raise NameError

    '''
    👇👇👇👇state[tls]结构解析👇👇👇👇
    ===============================================================
    state['e3'] = [[[[(x1,y1)],...,[(xn,yn)]],[[[(x1,y1)],[(x1,y1)],[(x1,y1)]]]]
                  ↑
                  defaultdict额外的中括号(squeeze(axis==0)去掉)
                  [[[[(x1,y1)],...,[(xn,yn)]],[[[(x1,y1)],[(x1,y1)],[(x1,y1)]]]]
                   ↑
                   e3 traffic light 的中括号，内容为e3管理的四条边
                  [[[[(x1,y1)],...,[(xn,yn)]],[[[(x1,y1)],[(x1,y1)],[(x1,y1)]]]]
                    ↑
                    lane2的中括号
                  [[[[(x1,y1)],...,[(xn,yn)]],[[[(x1,y1)],[(x1,y1)],[(x1,y1)]]]]
                     ↑
                     lane2_0的中括号
                  [[[[(x1,y1)],...,[(xn,yn)]],[[[(x1,y1)],[(x1,y1)],[(x1,y1)]]]]
                       ↑
                       lane2_0的x1,y1坐标点括号
    =================================================================
    state['e3'].append([1,2,3])
    defaultdict(<class 'list'>, {'e3': [[1, 2, 3]]}) 多了一个中括号，所以一共是四个中括号
    因此，
    state[i] = np.array(state[i]).squeeze(axis=0)的作用是：
    把defaultdict多的那层中括号[]去掉，因此变更后的state[i]为：
    [[[[x1,y1],...,[xn,yn]],[[[x1,y1],[x1,y1],[x1,y1]]]]
    变更后的state为： 
    state = {'e3':[[[[x1,y1],...,[xn,yn]],[[[x1,y1],[x1,y1],[x1,y1]]]],'e7':...,}
    另：
    np.array([()])会把元组()自动转成中括号[[]]
    ===============关于squeeze=============
    a
    Out[29]: 
    array([[[1],
            [2],
            [3]]])

    a.squeeze()
    Out[30]: array([1, 2, 3])

    a.squeeze(axis=0)
    Out[31]: 
    array([[1],
           [2],
           [3]])
    =======================================
    获取静态路网的网格坐标
    '''    
    def get_net_coords(self):
        '''
        获取网格的经纬度坐标
        '''
        # getShape: [(起始x, 起始y), (终止x，终止y)， 起始和终止均取中点]
        net_coords = defaultdict(dict) #该字典的key的默认value是list，可以添加元素
        
        
        for tls in self.tls_to_lane.keys():
            tls_encode = defaultdict(list)
            # 返回 2 * 3 * 60 * 4 grids
            for lane_id in self.tls_to_lane[tls]:
                 #二维矩阵 lane:[[lane_0],[lane_1],[lane_2]] lane_0:[(x1,y1),...,(xn,yn)]
                 #getshape:
                 #https://github.com/eclipse/sumo/blob/main/tools/sumolib/net/lane.py
                 #Returns the shape of the lane in 2d.
                 #This function returns the shape of the lane, as defined in the net.xml
                 #file. The returned shape is a list containing numerical
                 #2-tuples representing the x,y coordinates of the shape points.
                 #For includeJunctions=True the returned list will contain
                 #additionally the coords (x,y) of the fromNode of the
                 #corresponding edge as first element and the coords (x,y)
                 #of the toNode as last element.
                 #For internal lanes, includeJunctions is ignored and the unaltered
                 #shape of the lane is returned.
                 
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

        #print("state_old",state['e3'])
        #print("==========================================")
        #print(net_coords)
        
        #raise NameError

        #print(net_coords['D1']['D2D1_0'][0][0])

        #for tls in net_coords.keys():
        #    net_coords[tls] = np.array(net_coords[tls]).squeeze(axis=0)

        #print("state_new",state['e3']) #squeeze掉了元组的小括号
        #print("state_new",state) #squeeze掉了元组的小括号
        #print(net_coords)
        
        #print(np.array(net_coords['D1']).shape)

        return net_coords
    
    '''
    def get_size(self):
        tls_size = defaultdict(dict)
        for tls in self.net_coords.keys():
            lanes = defaultdict(tuple)
            for lane in self.net_coords[tls].keys():
                length = len(self.net_coords[tls][lane])
                lanes[lane]=(length,)
            tls_size[tls]=lanes
        return tls_size
    '''
    
    def build_maskarr(self,tls:str):
        lanes = dict()
        for lane in self.net_coords[tls].keys():
            length = len(self.net_coords[tls][lane])
            lanes.update({lane:np.zeros((length,))})
        return lanes