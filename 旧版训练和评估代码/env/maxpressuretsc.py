import random
from collections import deque

from .trafficsignalcontroller import TrafficSignalController


class MaxPressureTSC(TrafficSignalController):
    def __init__(self,
                 conn,
                 tsc_id,
                 netdata,
                 mode="deprecated",
                 red_t=3,
                 yellow_t=2,
                 green_t=5):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.green_t = green_t
        self.t = 0
        # for keeping track of vehicle counts for websters calc
        # print(tsc_id)
        self.phase_deque = deque()
        self.max_pressure_lanes = self.max_pressure_lanes()
        self.data = None
        # store how many green movements each phase has
        # for breaking ties in max pressure
        self.phase_g_count = {}
        for p in self.green_phases:
            self.phase_g_count[p] = sum([1 for m in p if m == 'g' or m == 'G'])

    def next_phase(self):
        ###need to do deque here
        if len(self.phase_deque) == 0:
            max_pressure_phase = self.max_pressure()
            phases = self.get_intermediate_phases(self.phase, max_pressure_phase)
            self.phase_deque.extend(phases + [max_pressure_phase])
        return self.phase_deque.popleft()

    def max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                inc_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.add(ol)

            max_pressure_lanes[g] = {'inc': inc_lanes, 'out': out_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        # compute pressure for all green movements
        for g in self.green_phases:
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            # pressure is defined as the number of vehicles in a lane
            inc_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
            # print("inc_pressure:", inc_pressure)
            out_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in out_lanes])
            # print("out_pressure:", out_pressure)
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)

        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(self.green_phases):
            return random.choice(self.green_phases)
        else:
            # choose phase with max pressure
            # if two phases have equivalent pressure
            # select one with more green movements
            # return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [(p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p: p[1], reverse=True)
            phase_pressure = [p for p in phase_pressure if p[1] == phase_pressure[0][1]]
            # return: 'GGGrrrrrGGGrrrrr'
            return random.choice(phase_pressure)[0]

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data):
        # print("=====> data: ", data)
        # print("\n\n")
        self.data = data
