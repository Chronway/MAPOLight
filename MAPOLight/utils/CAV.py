# -*- coding: utf-8 -*-
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

    def _iscontain(self, coord):
        self.pos = numpy.array(traci.vehicle.getPosition(self.vid))
        
        distance = numpy.linalg.norm(self.pos-numpy.array(coord))

        if distance <= self.radius:#distance            
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
