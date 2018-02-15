import numpy as np
import cProfile
import re
import math
import matplotlib.pyplot as plt
import sys
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering
from gym import spaces

from time import sleep

from kdrl.agent import DQNAgent
from kdrl.policy import *
from kdrl.memory import *
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout

import gym

_NUM_DISTANCE_SENSOR = 20
_DISTANCE_SENSOR_MAX_DISTANCE = 200

param1 = int(sys.argv[1])#hidden1
param2 = int(sys.argv[2])#hidden2
param3 = int(sys.argv[3])#num_action
param4 = float(sys.argv[4])#broken_sensor
param5 = int(sys.argv[5])#memory
param6 = int(sys.argv[6])#episode
param7 = int(sys.argv[7])#num_obstacles
param8 = int(sys.argv[8])#pos_obstacles
param_name = str(sys.argv[9])#filename

class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.collider_func = collider_func
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.abs_pos = np.copy(self.pos)
        self.abs_angle = self.angle
        self.trans = rendering.Transform()
        self.segments_cache = None
        #
        self.add_attr(self.trans)
    def render(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.geom.render()
    #
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update()
    def _move_by_xy(self, diff_x, diff_y):
        self.set_pos(self.pos[0] + diff_x, self.pos[1] + diff_y)
    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    def set_angle(self, angle, deg=True):
        self.angle = angle #if not deg else np.deg2rad(angle)
        self.update()
    def rotate(self, diff_angle, deg=True):
        self.set_angle(self.angle + diff_angle)
    def update(self):
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        #
        self.abs_pos[:] = 0
        self.abs_angle = 0
        prev_angle = 0
        for attr in reversed(self.attrs):
            if isinstance(attr, rendering.Transform):
                self.abs_pos += rotate(attr.translation, prev_angle)
                self.abs_angle += attr.rotation
                prev_angle = attr.rotation
        self.segments_cache = None
    def get_segments(self):
        if self.segments_cache is None:
            self.segments_cache = self.collider_func(self.abs_pos, self.abs_angle)
        return self.segments_cache
    def get_intersections(self, segment_list):
        if self.collider_func is None:
            return []
        intersections = []
        for collider_segment in self.get_segments():
            for segment in segment_list:
                intersection = collider_segment.get_intersection(segment)
                if intersection is not None:
                    intersections.append(intersection)
        return intersections
    def get_geom_list(self):
        return [self]

def get_nearest_point(pos_list, ref_pos):
    sorted_pos_list = sorted(pos_list, key=lambda pos: np.linalg.norm(pos - ref_pos, ord=2))
    return sorted_pos_list[0]

class Segment():
    def __init__(self, start=(0, 0), end=(0, 0)):
        self.start = np.asarray(start, dtype=np.float32)
        self.end = np.asarray(end, dtype=np.float32)
    def diff_x(self):
        return self.end[0] - self.start[0]
    def diff_y(self):
        return self.end[1] - self.start[1]
    def update_start_end(self, start, end):
        self.start[:] = start
        self.end[:] = end
    def get_intersection(self, segment):
        def check_intersection_ls(line, segment):
            l = line.end - line.start
            p1 = segment.start - line.start
            p2 = segment.end - line.start
            return (p1[0]*l[1] - p1[1]*l[0] > 0) ^ (p2[0]*l[1] - p2[1]*l[0] > 0) # TODO: sign==0
        def check_intersection_ss(seg1, seg2):
            return check_intersection_ls(line=seg1, segment=seg2) and check_intersection_ls(line=seg2, segment=seg1)
        s1, s2 = self, segment
        if check_intersection_ss(s1, s2):
            r = (s2.diff_y() * (s2.start[0] - s1.start[0]) - s2.diff_x() * (s2.start[1] - s1.start[1])) / (s1.diff_x() * s2.diff_y() - s1.diff_y() * s2.diff_x())
            return (1 -r) * s1.start + r * s1.end
        else:
            return None

class Wall(GeomContainer):
    def __init__(self, start, end, color, **kwargs):
        GeomContainer.__init__(self, rendering.Line(start, end), collider_func=self.collider_func, **kwargs)
        self.set_color(*color)
        self.wall_segment = Segment(start, end)
    def set_pos(self, pos_x, pos_y):
        pass
    def set_angle(self, angle, deg=False):
        pass
    def collider_func(self, *args):
        return [self.wall_segment]
    

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
        self.value = None
    def detect(self, obstacles):
        raise NotImplementedError()
    def _set_sensor_value(self, value):
        self.value = value

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.ray_geom.set_color(1, 0.5, 0.5)
        self.effect_geom = GeomContainer(rendering.make_circle(radius=5, filled=False))
        self.effect_geom.set_color(1, 0.5, 0.5)
        self.intersection_pos = [0, 0]
        self.max_distance = _DISTANCE_SENSOR_MAX_DISTANCE
        self._ray_segment = Segment()
        self._update_ray_segment()
    def render(self):
        Sensor.render(self)
        self.ray_geom.start = self.abs_pos
        self.ray_geom.end = self.intersection_pos
        self.ray_geom.render()
        self.effect_geom.set_pos(*self.intersection_pos)
        self.effect_geom.render()
    def get_geom_list(self):
        return Sensor.get_geom_list(self) + [self.ray_geom]
    def _update_ray_segment(self):
        self._ray_segment.update_start_end(self.abs_pos, self.abs_pos + rotate([self.max_distance, 0], self.abs_angle))
    def detect(self, obstacles):
        self._update_ray_segment()
        intersections = []
        for obs in obstacles:
            intersections += obs.get_intersections([self._ray_segment])
        if len(intersections) > 0:
            intersection_pos = get_nearest_point(intersections, self.abs_pos)
            distance = np.linalg.norm(intersection_pos - self.abs_pos, ord=2)
        else:
            intersection_pos = self._ray_segment.end
            distance = self.max_distance
        self.intersection_pos = intersection_pos
        self._set_sensor_value(distance/_DISTANCE_SENSOR_MAX_DISTANCE)
    
                


class Robot(GeomContainer):
    def __init__(self, **kwargs):
        geom = rendering.make_circle(30)
        collider_func = None
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        #
        self.set_color(0, 0, 1)
        #
        self.sensors = []
        for i in range(_NUM_DISTANCE_SENSOR):
            dist_sensor = DistanceSensor(rendering.make_circle(5))
            if i == 0:
                dist_sensor.set_color(0, 0, 0)
            else:
                dist_sensor.set_color(1, 0, 0)
            dist_sensor.set_pos((rotate([30, 0], 360 / 20 * i, deg=True)))
            dist_sensor.set_angle(360 / 20 * i, deg=True)
            dist_sensor.add_attr(self.trans)
            self.sensors.append(dist_sensor)
    def render(self):
        GeomContainer.render(self)
        for sensor in self.sensors:
            sensor.render()
    def get_geom_list(self):
        return GeomContainer.get_geom_list(self) + self.sensors
    def update(self):
        GeomContainer.update(self)
        for sensor in self.sensors:
            sensor.update()
    def update_sensors(self, visible_objects):
        for sensor in self.sensors:
            sensor.detect(visible_objects)
    def get_sensor_values(self):
        return [sensor.value for sensor in self.sensors]

UNIT_SQUARE = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2

def rotate(pos_array, angle, deg=True):
    pos_array = np.asarray(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.asarray([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T

def polyline_to_segmentlist(polyline):
    return [Segment(polyline[i - 1], polyline[i]) for i in range(len(polyline))]


class ObstacleEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 400
        self._seed()
        self.state = np.zeros(_NUM_DISTANCE_SENSOR + 3, dtype=np.float32)
        self.viewer = None
        self.robot = Robot()
        self.obstacles = []
        for i in range(param7):
            obs = GeomContainer(rendering.make_polygon(UNIT_SQUARE * 50), lambda pos, angle: polyline_to_segmentlist(rotate(UNIT_SQUARE, angle) * 50 + pos))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
        self.obstacles.append(Wall([0, 0], [self.screen_width, 0], (0, 1, 0)))
        #self.obstacles.append(Wall([self.screen_width, 0], [self.screen_width, self.screen_height], (0, 1, 0)))
        self.obstacles.append(Wall([self.screen_width, self.screen_height], [0, self.screen_height], (0, 1, 0)))
        self.obstacles.append(Wall([0, self.screen_height], [0, 0], (0, 1, 0)))
        #
        self.visible_object = []
        self.register_visible_object(self.robot)
        for obs in self.obstacles:
            self.register_visible_object(obs)
    def _step(self, action):
        if action == 0:
            self.robot.set_angle(0)
            self.robot.move(2)
        elif action == 1:
            self.robot.set_angle(np.deg2rad(60))
            self.robot.move(2)
        elif action == 2:
            self.robot.set_angle(np.deg2rad(-60))
            self.robot.move(2)
        elif action == 3:
            self.robot.set_angle(np.deg2rad(30))
            self.robot.move(2)
        elif action == 4:
            self.robot.set_angle(np.deg2rad(-30))
            self.robot.move(2)
        self.update_state()
        #

        done = (min(self.state[3:_NUM_DISTANCE_SENSOR + 3]) <= 0.02) \
            or (self.robot.pos[0] > 600)

        if min(self.state[3:_NUM_DISTANCE_SENSOR + 3]) <= 0.02:
            reward = -1000
        elif self.robot.pos[0] > 600:
            reward = 1000
        else:
            rew =0
            for sen in range(_NUM_DISTANCE_SENSOR):
                rew += pow(self.robot.get_sensor_values()[sen],2)
            reward = rew/20
            

        return self.state, reward, done, {}

    def _reset(self):
        self.count_step = 0
        self.robot.set_pos(100, self.screen_height/2)
        self.robot.set_angle(0)
        if param8 == 0:
            self.obstacles[0].set_pos(200, self.screen_height/2 + 100)
            self.obstacles[1].set_pos(200, self.screen_height/2 - 100)
            self.obstacles[2].set_pos(350, self.screen_height/2)
            self.obstacles[3].set_pos(500, self.screen_height/2 + 100)
            self.obstacles[4].set_pos(500, self.screen_height/2 - 100)
            for obs in self.obstacles:
                obs.set_angle(0)        
        else:    
            for obs in self.obstacles:
                obs.set_pos(randint(200, 600), randint(0, self.screen_height))
                obs.set_angle(0)
        self.update_state()
        return self.state
    def update_state(self):
        self.count_step += 1
        self.robot.update_sensors(self.visible_object)
        self.state[0:2] = self.robot.pos.tolist()
        self.state[2:3] =np.array(self.robot.angle, dtype="float32")
        ###broken_sensor###
        if param4 == 1:
            #self.state[3:4] = 1
            #self.state[4:_NUM_DISTANCE_SENSOR + 3] = self.robot.get_sensor_values()[1:20]
            #self.state[_NUM_DISTANCE_SENSOR + 1:_NUM_DISTANCE_SENSOR + 3] = np.ones(2, dtype=np.float32)
            broken = randint(1, 4)
            if self.count_step % 2 == 1:
                self.state[3:_NUM_DISTANCE_SENSOR + 3] = self.robot.get_sensor_values()
            else:
                self.state[3:_NUM_DISTANCE_SENSOR + 3] = np.ones(_NUM_DISTANCE_SENSOR, dtype=np.float32)
        
        else:
            self.state[3:_NUM_DISTANCE_SENSOR + 3] = self.robot.get_sensor_values()
        
    def register_visible_object(self, geom_container):
        self.visible_object.extend(geom_container.get_geom_list())
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            self.st = rendering.Line((100,0), (100,self.screen_height))
            self.st.set_color(0,0,0)
            self.viewer.add_geom(self.st)            
            self.goal = rendering.Line((600,0), (600,self.screen_height))
            self.goal.set_color(0,0,0)
            self.viewer.add_geom(self.goal)
            #
            for geom in self.visible_object:
                self.viewer.add_geom(geom)
        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))


def get_model(state_shape, num_actions):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(param1, activation='relu'),
                       Dense(param2, activation='relu'),          
                       Dense(num_actions)])

def main():
    from gym.envs.registration import register
    register(
        id='Obstacle-v0',
        entry_point=ObstacleEnv,
        max_episode_steps=600,
        reward_threshold=100.0,
        )
    env = gym.make('Obstacle-v0')
    #
    state_shape = (_NUM_DISTANCE_SENSOR + 3,)
    num_actions = param3

    agent = DQNAgent(core_model=get_model(state_shape, num_actions),
                     num_actions=num_actions,
                     optimizer='adam',
                     policy=Boltzmann(),
                     memory=param5,
                     )
    # training
    goal = []
    episodes = param6
    for episode in range(episodes):
        state = env.reset()
        reward_sum = 0
        #print(episode)
        action = agent.start_episode(state)
        while True:
            #env.render()
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if not done:
                action = agent.step(state, reward)
                continue
            else:
                agent.end_episode(state, reward)
                break
    ###print_graph
        if reward >= 100:
            goal.append(1.0)
            #print("o")   
        else:
            goal.append(0.0)
            #print("x")
        if episode % 1000 == 0:
                print("episode: {}/{}, score: {}"
                        .format(episode, episodes, reward_sum))
    n = 100
    v = np.ones(n, dtype="float32")/n
    goal_sma = np.convolve(goal, v, mode='valid')
    epi = np.arange(episodes - n + 1)
    plt.xlim(xmin=0, xmax=param6)
    plt.ylim(ymin=0, ymax=0.5)
    plt.grid()
    plt.plot(epi, goal_sma)
    #plt.legend()
    filename = "graph_{param}.png".format(param=param_name)
    plt.savefig(filename)
    ###
    csvname = "csvtrane_{param}.csv".format(param=param_name)
    csv = np.c_[np.arange(param6),goal]
    np.savetxt(csvname,csv,delimiter=',')

    # test
    goal = []
    for episode in range(5):
        state = env.reset()
        reward_sum = 0
        while True:
            #env.render()
            action = agent.select_best_action(state)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
        print('episode {} score: {}'.format(episode, reward_sum))
       
    
          
if __name__ == '__main__':
    main()