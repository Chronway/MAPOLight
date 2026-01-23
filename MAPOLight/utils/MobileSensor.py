# -*- coding: utf-8 -*-

# import os
import traci
import numpy as np
# from numba import jit
# import sumolib
from math import floor
# from traci import vehicle
# from traci import TraCIException
from .Network import Network
from .CAV import CAV

# import Network
# import CAV

CELL_NUM = 60
CELL_LENGTH = 5
# LANE_LENGTH = 300
MAX_SPEED = 50


class CAVs:
    def __init__(self, PR, netfile="", debug=False):
        self.PR = PR  # penetration rate
        self.running_vids = set()  # all runing vehicles id set
        self.cavids = set()  # cav id set
        self.cavs = set()  # cav objects set
        self.detected_vids = set()  # cav detected vehicles set
        if debug:
            self.network = Network(netfile)
        # self.tls_state = None

    def get_allvids(self):
        running_vechicles = traci.vehicle.getIDList()
        return set(running_vechicles)

    def get_cavs(self):
        step_cavs = set()
        insert_vehs = np.array(traci.simulation.getDepartedIDList())
        rnd = np.random.rand(len(insert_vehs))
        mask = rnd < self.PR
        # print(mask)
        new_vids = insert_vehs[mask]
        # print("new_vehs:",insert_vehs)
        # print(len(new_vehs)/(len(insert_vehs)+1))

        cavids = self.cavids & self.running_vids | set(new_vids)

        for cavid in cavids:
            step_cavs.add(CAV(vid=cavid, radius=40))
        return cavids, step_cavs  # first is id of cavs, second are cav objects

    def get_detectvids(self):
        search_area = self.running_vids
        detected_vids = set()
        for cav in self.cavs:
            detected = cav.get_detected(search_area)
            detected_vids = detected_vids | detected
            search_area = search_area - detected  # reduce search range to accelerate search, optional
        return detected_vids

    # update all observations,matain
    # 1.total vehicles
    # 2.cavs
    # 3.detected vehicles
    def update_observations(self):
        self.running_vids = self.get_allvids()
        self.cavids, self.cavs = self.get_cavs()
    def update_detects(self):
        self.detected_vids = self.get_detectvids()

    @property
    def devids(self):
        return self.detected_vids

    def veh_coverage(self):
        return len(self.detected_vids) / len(self.running_vids)

    def cav_coverage(self):
        return len(self.cavids) / len(self.running_vids)

    def __len__(self):
        return len(self.cavids)

    # maintain()->get_state()
    def update(self):
        self.update_observations()
        self.tls_state = self.get_mask()

    def __contain(self, coord):
        for cav in self.cavs:
            # if cav.vid in exist_vehs:
            if cav._iscontain(coord):
                # print("exist!")
                return True
            # else:
            # print("not exist!!!!!!!!!")
        return False

    def get_mask(self):
        tls_mask = dict()
        network_coords = self.network.net_coords
        for tls in network_coords.keys():
            mask = self.network.build_maskarr(tls)
            # print(mask)
            for lane in network_coords[tls]:
                cell_num = len(network_coords[tls][lane])
                for pos in range(cell_num):
                    if self.__contain(network_coords[tls][lane][pos]):
                        mask[lane][pos] = 1
            one_dim = self.mask2array(mask)
            # print(type(one_dim))
            # print("???????????????????????")
            # print(one_dim.shape)
            tls_mask.update({tls: one_dim})
        return tls_mask

    def coverage(self):
        count = 0
        total = 0
        for tls in self.tls_state.keys():
            masks = self.tls_state[tls]
            # print("[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]")
            # print(type(masks))
            elements_1 = np.sum(masks == 1)
            count += elements_1
            total += masks.shape[0]
        # print(len(self.tls_state.keys()) * self.tls_state['e3'].shape[0] * self.tls_state['e3'].shape[1])
        # return count / (len(self.tls_state.keys()) * self.tls_state['e3'].shape[0] * self.tls_state['e3'].shape[1])
        return count / total

    def mask2array(self, mask: dict):
        one_dim = np.array(list())
        for lane_mask in mask.values():
            one_dim = np.append(one_dim, lane_mask)
        # print("||||||||||||||||||||||||||||||||||||")
        # print(one_dim)
        return np.array(one_dim)

    def clear_cavs(self):
        self.cavs.clear()


if __name__ == "__main__":
    cav = CAVs(0.1)
    cav.update()
    cav.get_state()
    # print(cav.raw_state)
