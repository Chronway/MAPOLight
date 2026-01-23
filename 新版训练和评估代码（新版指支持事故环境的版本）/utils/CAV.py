# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:20:30 2022

@author: Wang Chong
"""
import traci
import numpy 
#from traci import TraCIException

class CAV:

    def __init__(self, vid, radius = 40):
        self.vid = vid
        self.radius = radius
    
    @property
    def id(self):
        return self.vid

    def _iscontain(self, coord): #单车是contains
        '''
        判断坐标是否在车辆的探测范围内
        :param coord: 待判断坐标
        :return: boolean, 在探测范围内返回true
        '''
        #coord_x, coord_y = coord
        #try:
        #    self.x_pos, self.y_pos = traci.vehicle.getPosition(self.vid) #vhicle.getposition id是很重要的参数，需要获得该车的位置，而不是随意命名一个index的车辆id的位置
        #except TraCIException as e: #防止vid找不到
        #    print("_iscontain:",str(e))
        #    return False
        
        #self.x_pos, self.y_pos = traci.vehicle.getPosition(self.vid) #vhicle.getposition id是很重要的参数，需要获得该车的位置，而不是随意命名一个index的车辆id的位置
        self.pos = numpy.array(traci.vehicle.getPosition(self.vid))
        
        #print("c",coord)
        #print("rc:",(self.x_pos,self.y_pos))
        #if (coord_x - self.x_pos)*(coord_x - self.x_pos) + (coord_y - self.y_pos)*(coord_y - self.y_pos) <= self.radius*self.radius:
        distance = numpy.linalg.norm(self.pos-numpy.array(coord))

        if distance <= self.radius:#distance
            #print("r*r:",self.radius*self.radius)
            #print("veh coordx:",coord_x,"veh coordy:",coord_y)
            #print("self coordx:",self.x_pos,"self coordy:",self.y_pos)
            #print("=========================")
            
            #print("========================")
            #print(self.pos)
            #print(coord)
            #print(distance)
            #print(self.radius)
            #print("-----------------------")
            
            return True
        else:
            return False
    
    def get_detected(self,
                    search_vehs:set # search area that looking for detected vehicles
                    ):
        detected = set()
        for veh in search_vehs:
            pos = traci.vehicle.getPosition(veh)
            if(self._iscontain(pos)):
                detected.add(veh)
        #detected.add(self.vid) #self count
        return detected