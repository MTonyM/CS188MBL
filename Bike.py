import simpy
import random
import math
import numpy as np

import Scheduler
import Visualize

ALL_STATIONS = {}

NUM_STATIONS = 12
INIT_NUM = 100
NUM_BIKES = NUM_STATIONS * INIT_NUM
SLICES = 72
TIME = 14500

CALL_SCHEDULER = []

SAMPLES = []
REWARDS = []
ALL_FLOWS = []
TOTAL_REWARDS_EACHDAY = []

algo = Scheduler.Scheduler(NUM_BIKES, NUM_STATIONS, SLICES, 0.9, 1e-9)


def init_samples():
    if len(SAMPLES) == 0:
        for i in range(NUM_STATIONS):
            SAMPLES.append([])
            for _ in range(SLICES):
                SAMPLES[i].append(0)
    else:
        for i in range(NUM_STATIONS):
            for j in range(SLICES):
                SAMPLES[i][j] = 0


def init_rewards():
    if len(REWARDS) == 0:
        for _ in range(NUM_STATIONS):
            REWARDS.append(0)
    else:
        for i in range(NUM_STATIONS):
            REWARDS[i] = 0


def init_allflows():
    if len(ALL_FLOWS) == 0:
        for i in range(SLICES):
            ALL_FLOWS.append([])
            for j in range(NUM_STATIONS):
                ALL_FLOWS[i].append([])
                for _ in range(NUM_STATIONS):
                    ALL_FLOWS[i][j].append(0)
    else:
        for i in range(SLICES):
            for j in range(NUM_STATIONS):
                for k in range(NUM_STATIONS):
                    ALL_FLOWS[i][j][k] = 0


def init_caller(env):
    for i in range(NUM_STATIONS):
        CALL_SCHEDULER[i] = env.event()


def record_total_rewards():
    reward = 0
    for r in REWARDS:
        reward += r
    TOTAL_REWARDS_EACHDAY.append(reward)


def record_json(rewards, met):
    f = open("data\\" + "rewards_" + met + "_" + "sche.json", 'w')

    print("{", file = f)

    print('  "Rewards":[', end = "", file = f)
    for i in range(len(rewards) - 1):
        print(str(rewards[i]), end = ",", file = f)
    print(str(rewards[len(rewards) - 1]) + "],", file = f)

    print('  "Day":[', end = "", file = f)
    for i in range(len(rewards) - 1):
        print(str(i + 1), end = ",", file = f)
    print(str(len(rewards)) + "],", file = f)

    print('  "categ":[', end = "", file = f)
    for _ in range(len(rewards) - 1):
        print('"' + met + '"', end = ",", file = f)
    print('"' + met + '"' + "]", file = f)

    print("}", file = f)

    f.close()


def out_distri_uniform(a, b):
    return random.randint(a, b)


def normalize(distribution):
    s = 0
    nDistribution = {}
    for location in distribution.keys():
        s += distribution[location]
    for location in distribution.keys():
        nDistribution[location] = distribution[location] / s
    return nDistribution


def getStationFromIndex(idx):
    return ALL_STATIONS[idx]


def computeTransitionTime(start, end):
    distance = computeDistance(start, end)

    ''' The distribution would be modified upon hypothesis '''
    lower = min(1, distance - 2)
    # return out_distri_uniform(lower, distance + 1) + 0.5
    return 0.5
    ''' --- --- --- --- --- --- --- --- --- --- --- --- -- '''


def computeDistance(start, end):
    xs, sy = start.getPosition()
    es, ey = end.getPosition()
    return abs(xs - es) + abs(sy - ey)


def generateDispatcherDistribution(start):
    start_id = start.getIndex()
    distribution = {}
    for station in ALL_STATIONS.values():
        end_id = station.getIndex()

        # Not going to the start station
        if start_id == end_id:
            continue

        ''' The distribution would be modified upon hypothesis '''
        distance = computeDistance(start, station)
        distribution[end_id] = math.exp(-0.2 * distance)
        # distribution[end_id] = math.exp(-0.5 * 3)
        ''' --- --- --- --- --- --- --- --- --- --- --- --- -- '''

    # Normalize the distribution
    distribution = normalize(distribution)
    return distribution



def generateDispatcherNumbers(start, nBikes):
    if start.dispatch_distribution == None:
        start.dispatch_distribution = generateDispatcherDistribution(start)

    # generate number of bikes using sampling
    numbers = {}
    for _ in range(nBikes):
        sample = random.random()
        possiblility = 0
        for location in start.dispatch_distribution.keys():
            possiblility += start.dispatch_distribution[location]
            if sample < possiblility:
                # Add a sample
                if location in numbers.keys():
                    numbers[location] += 1
                else:
                    numbers[location] = 1
                break

    return numbers


class BikeScheduler:
    def __init__(self, env):
        self.env = env
        self.process = env.process(self.scheduler())

    def bikeScheduler(self, flows, remains, rewards):
        # 1 Do nothing for scheduling
        schedules = []
        for i in range(NUM_STATIONS):
            schedules.append(INIT_NUM)

        # 2 Simple naive greedy method
        # schedules = algo.naive_scheduler(remains, rewards)
        # print(schedules)

        # 3 Reinforcement Learning
        # print("start")
        # schedules = algo.greedy_scheduler2(np.ceil(flows), np.array(rewards))
        # print("check outside schedules")
        # print(schedules)

        return schedules # A set of how much bikes for each station

    def scheduler(self):
        # The process scheduling the bikes at the end of the day
        while True:
            # Wait for call schedulers
            yield simpy.events.AllOf(self.env, CALL_SCHEDULER)
            init_caller(self.env) # Recreate caller triggers
            # print("DAY")

            # Record the total rewards at the end of the day
            record_total_rewards()

            # Call scheduler's algorithm
            # print("ready to schedule")
            print("check input")
            print(np.array(ALL_FLOWS[0]))
            # print(ALL_FLOWS[1])

            # print(ALL_FLOWS[71])
            schedules = list(self.bikeScheduler(ALL_FLOWS, SAMPLES, REWARDS))
            # print(schedules)

            # Init samples and rewards
            init_samples()
            init_rewards()
            init_allflows()

            # Do scheduling
            if len(schedules) != NUM_STATIONS:
                pass
            else:
                for i in range(NUM_STATIONS):
                    s = getStationFromIndex(i)
                    if s.bikes.level != 0:
                        yield s.bikes.get(s.bikes.level)
                    if schedules[i] != 0:
                        yield s.bikes.put(int(schedules[i]))
                    # print("check putting")
                    # print(s.bikes.level)

class Buffer:
    def __init__(self):
        self.buffer = []

    def push(self, elem):
        self.buffer.append(elem)

    def pop(self):
        assert(len(self.buffer) != 0)
        return self.buffer.pop(0)

    def isNULL(self):
        return len(self.buffer) == 0

class Map:
    def __init__(self, env):
        self.env = env
        for i in range(NUM_STATIONS):
            pos_x = i
            pos_y = i + 1

            ''' Initial Number '''
            init_bike = INIT_NUM
            # if i == 1:
            #     init_bike = NUM_BIKES
            # else:
            #     init_bike = 0
            '''--- --- --- --- '''

            ALL_STATIONS[i] = Station(env, (pos_x, pos_y), i, init_bike)
            CALL_SCHEDULER.append(env.event())

class Station:
    def __init__(self, env, position, idx, initial):
        self.env = env
        self.position = position # Position should be a tuple (x, y)
        self.idx = idx
        self.bikes = simpy.Container(env, init = initial)
        self.buf = Buffer()
        self.dispatch_distribution = None
        self.slice = 0
        self.day = 0

        ''' The distribution would be modified upon hypothesis '''
        self.mean = out_distri_uniform(10, 30)
        ''' --- --- --- --- --- --- --- --- --- --- --- --- -- '''

        self.process = env.process(self.run())
        self.dispatcher = env.process(self.dispatcher())
        self.going = env.event()

    def run(self):
        while True:
            # print("a new day: %d", self.day)
            # Running one day of bike riding
            yield self.env.process(self.one_day())
            # print("one day finish: %d", self.day)

            # Make sure the last dispatch of day finished
            while not self.buf.isNULL:
                yield self.env.timeout(20)
            # print("last dispatch finish: %d", self.day)

            # Tell scheduler that this station is ready
            CALL_SCHEDULER[self.getIndex()].succeed()
            # print("call to schedule: %d", self.day)
            yield self.env.timeout(20)
            self.day += 1

    def dispatcher(self):
        while True:
            yield self.going

            scheme = generateDispatcherNumbers(self, self.buf.pop())
            # print("check scheme")
            # print(scheme)
            preorder = {}
            for sid in scheme.keys():
                preorder[sid] = computeTransitionTime(self, ALL_STATIONS[sid])

            # Dispatch bikes
            order = sorted(preorder.items(), key=lambda item:item[1])
            time = 0
            for dest in order:
                yield self.env.timeout(dest[1] - time)
                time = dest[1]
                s = getStationFromIndex(dest[0])
                # print("check records")
                # print(scheme[dest[0]])
                # print(self.idx, s.getIndex(), scheme[dest[0]])
                ALL_FLOWS[self.slice][self.idx][s.getIndex()] = scheme[dest[0]]
                # print("check ALL_FLOWS")
                # print(ALL_FLOWS[self.slice][self.idx][s.getIndex()])
                yield s.bikes.put(scheme[dest[0]])
            # print(self.idx)
            # print("results", self.slice)
            # print(np.array(ALL_FLOWS[self.slice]))

    def one_day(self):
        for i in range(SLICES):
            self.slice = i
            SAMPLES[self.idx][i] = self.bikes.level # Record samples
            # Going out
            # print("time: " + str(self.env.now))
            # print(str(self.f)+": "+str(self.bikes.level))

            ''' The distribution would be modified upon hypothesis '''
            # outBike = out_distri_uniform(1, 20)
            outBike = out_distri_uniform(self.mean - 5, self.mean + 5) * 0+1
            if self.idx % 2 == 0 and i <= 36:
                outBike = out_distri_uniform(self.mean * 10 - 5, self.mean * 10 + 5)
            if self.idx % 2 == 1 and i > 36:
                outBike = out_distri_uniform(self.mean * 10 - 5, self.mean * 10 + 5)
            # print(outBike)
            # if self.idx > 4:
            #     outBike = 1
            ''' --- --- --- --- --- --- --- --- --- --- --- --- -- '''

            if outBike <= self.bikes.level:
                yield self.bikes.get(outBike)
            else:
                if self.bikes.level != 0:
                    outBike = self.bikes.level
                    yield self.bikes.get(outBike)
                else:
                    outBike = 0

            # print("time: " + str(self.env.now))
            # print(str(self.idx)+": "+str(self.bikes.level))
            # print("check needs")
            # print(outBike)
            REWARDS[self.idx] += outBike # Record usage as rewards

            # Dispatcher
            self.buf.push(outBike)
            self.going.succeed()
            self.going = self.env.event()

            # State unit time
            yield self.env.timeout(10)

    def getPosition(self):
        return self.position

    def getIndex(self):
        return self.idx

def main():
    init_samples()
    init_rewards()
    init_allflows()

    env = simpy.Environment()
    Map(env)
    BikeScheduler(env)

    env.run(until=TIME)

    print(TOTAL_REWARDS_EACHDAY)
    # record_json(TOTAL_REWARDS_EACHDAY, "no")
    record_json(TOTAL_REWARDS_EACHDAY, "naive")
    # record_json(TOTAL_REWARDS_EACHDAY, "rl")

if __name__ == '__main__':
    main()