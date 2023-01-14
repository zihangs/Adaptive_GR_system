import gym
from gym import spaces
from stable_baselines3 import A2C, PPO


import os
import re
import subprocess
import numpy as np
import pandas as pd
import statistics
import shutil


def func_precision(stringList, answer):
    goal_count = 0
    found = 0
    for result in stringList:
        if result == str(answer):
            found = 1
        goal_count += 1

    return found/(goal_count-1)

def func_recall(stringList, answer):
    found = 0
    for result in stringList:
        if result == str(answer):
            found = 1
            break
    return found

def func_accuracy(total, stringList, answer):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in stringList[0:-1]:
        if result == str(answer):
            tp += 1
        else:
            fp += 1
    
    fn = 1 - tp
    
    # total is the number of all goals
    tn = total - tp - fp - fn
    return (tp + tn)/(tn + tp + fp + fn)


def func_bacc(total, stringList, answer):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in stringList[0:-1]:
        if result == str(answer):
            tp += 1
        else:
            fp += 1
    
    fn = 1 - tp
    
    # total is the number of all goals
    tn = total - tp - fp - fn

    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    bacc = (tpr + tnr)/2

    return bacc


# return a list of each statistics for every testing case
def calculate_statistics(rows):
    length = rows.shape[0]

    precision = []
    recall = []
    accuracy = []
    b_accuracy = []
        
    for index, row in rows.iterrows():
        
        answer = row["Real_Goal"]
        results = row["Results"].split("/")
        all_candidates = row["Cost"].split("/")
        
        total = len(all_candidates)-1   # the last one is /
        
        p = func_precision(results, answer)
        r = func_recall(results, answer)
        a = func_accuracy(total, results, answer)
        bacc = func_bacc(total, results, answer)
        
        precision.append(p)
        recall.append(r)
        accuracy.append(a)
        b_accuracy.append(bacc)
    
    return precision, recall, accuracy, b_accuracy


# a data point of all goal candidates
def averagedDataPoint(rows, goals):
    length = rows.shape[0]

    precision = []
    recall = []
    accuracy = []
    b_accuracy = []

    std_bacc = []

    tmp_precision = []
    tmp_recall = []
    tmp_accuracy = []
    tmp_b_accuracy = []
        
    for index, row in rows.iterrows():
        
        answer = row["Real_Goal"]
        results = row["Results"].split("/")
        all_candidates = row["Cost"].split("/")
        
        total = len(all_candidates)-1   # the last one is /
        
        p = func_precision(results, answer)
        r = func_recall(results, answer)
        a = func_accuracy(total, results, answer)
        bacc = func_bacc(total, results, answer)

        tmp_precision.append(p)
        tmp_recall.append(r)
        tmp_accuracy.append(a)
        tmp_b_accuracy.append(bacc)

        if len(tmp_b_accuracy) == goals:
            precision.append(statistics.mean(tmp_precision))
            recall.append(statistics.mean(tmp_recall))
            accuracy.append(statistics.mean(tmp_accuracy))
            b_accuracy.append(statistics.mean(tmp_b_accuracy))
            std_bacc.append(statistics.stdev(tmp_b_accuracy))

            tmp_precision = []
            tmp_recall = []
            tmp_accuracy = []
            tmp_b_accuracy = []
    
    return precision, recall, accuracy, b_accuracy, std_bacc


def tailAverage(metric_list, num):
    length = len(metric_list)
    if length < num:
        return statistics.mean(metric_list[0 : length])
    else:
        return statistics.mean(metric_list[(length - num) : length])


#### find the last number appears in the string and sort by it
def sortByLastNumber(lst):
    tupleList = []
    for item in lst:
        if item == ".DS_Store":
            continue
        numbers = re.findall(r'[\d]+', item)
        tupleList.append((int(numbers[-1]), item))  # sort by the last number
    tupleList.sort()
    
    stringList = []
    for item in tupleList:
        stringList.append(item[1])
    return stringList



###################################### rl ########################################

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, init_models, testing_data): # arg1, arg2, ...
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(10, ), dtype=np.float32)

        self.init_models = init_models
        self.testing_data = testing_data

        ## delete models and re-create it
        os.system("rm -rf ./tmp_models")
        os.mkdir("./tmp_models")
        self.models = "./tmp_models"

        ## copy models to ./tmp_models
        os.system("cp %s/*.pnml %s" % (init_models, self.models) )

        ## testing traces in drift (sort it)
        test_list = os.listdir(testing_data)
        self.test_list = sortByLastNumber(test_list)

        # fixed variables:
        self.recognizerJar = "./recognizer.jar"
        self.controllerJar = "./controller.jar"
        self.relearn_dir = "./Feedback/Add"
        self.num2relearn = 10
        self.numModels = self.countModels()

        self.totalCases = len(test_list)  # have .DS_store
        self.case_index = 0
        self.output_stats = "mytest.csv"
        self.averBACC = 1

    def countModels(self):
        models = 0
        for item in os.listdir(self.models):
            if item[-5::] == ".pnml":
                models += 1
        return models

    def allPositive(self, lst):
        for num in lst:
            if num < 0:
                return False
        return True

    def checkIfRemine(self):
        flag = "no"
        # # re-mine and replace models (in tmp_models) -> self.models
        for file in os.listdir():   # this converted xes are in the root directory
            if os.path.isfile(file) and file.split(".")[1] == "xes":
                
                # relearnTriggers.add(case_index)
                
                os.system("java -cp miner.jar autoMiner -DFM %s %s.pnml 0.8" % (file, file))
                os.system("rm %s" % file)
                os.system("mv %s.pnml %s" % (file, self.models) )
                flag = "yes"

        return flag

    def recentNCases(self):   # N = num2relearn
        for a_dir in os.listdir(self.relearn_dir):
            if a_dir == ".DS_Store":
                continue
                
            plans = os.listdir(self.relearn_dir + "/" + a_dir)
            plans = sortByLastNumber(plans)

            numOfPlans = len(plans)
            if numOfPlans > self.num2relearn:
                for i in range(1, numOfPlans):
                    former = self.relearn_dir + "/" + a_dir + "/sas_plan." + str(i)
                    latter = self.relearn_dir + "/" + a_dir + "/sas_plan." + str(i+1)
                    os.system("mv %s %s" % (latter, former) )

    def addRelearnFlag(self, filename, relearnFlag):
        with open(filename, 'rb+') as filehandle:
            filehandle.seek(-1, os.SEEK_END)
            filehandle.truncate()

        with open(filename, 'a') as filehandle:
            filehandle.write(',' + relearnFlag + '\n')
        
    # obs: 10 points,  # reward: if acc increase or not
    def step(self, action):  # action, relearn or not
        # update model
        controllerJar = "./controller.jar"
        recognizerJar = "./recognizer.jar"
        option = "-closedloop_ave_metric"  ## ??????  need to change the java code later
        relearn_dir = "./Feedback/Add"
        obs_percentage = 0.5   
        
        print("action: " + str(action))
        if action == 0 and self.allPositive(self.observation):
            # need to re-learn
            current_acc = 0.2
            acc_threshold = 0.8  # means to relearn

            os.system("java -jar %s %s %s %s %s" % (controllerJar, option, relearn_dir, current_acc, acc_threshold) )
            
            # remine model or not?
            relearnFlag = self.checkIfRemine()
            # add flag
            self.addRelearnFlag(self.output_stats, relearnFlag)

            # do another 10 gr
            for i in range(self.num2relearn*self.numModels):
                test_plan = self.test_list[self.case_index]
                real_goal = re.findall(r'[\d]+', test_plan)[-2]
                test_plan = self.testing_data + "/" + test_plan

                os.popen("java -cp %s Recognizer -w %s %s %s %s %s %s %s %s %s" % 
                    (recognizerJar, self.models, test_plan, real_goal, obs_percentage, 50, 2.5, 2.1, 0.8, self.output_stats)).read()

                self.case_index += 1
                print(self.case_index)
                self.recentNCases()

                if self.case_index == self.totalCases:
                    break

            # self.observation = next 10 recent bacc
            data = pd.read_csv("./%s" % self.output_stats, usecols=[0,1,2,3,4])
            p, r, a, bacc, std_bacc = averagedDataPoint(data, self.numModels)

            self.observation = bacc[-self.num2relearn::]
            newAverBACC = tailAverage(bacc, self.num2relearn)
            self.reward = newAverBACC - self.averBACC
            self.averBACC = newAverBACC

        else:
        # if action == 1:
            # do next 1 gr
            for i in range(self.numModels):
                test_plan = self.test_list[self.case_index]
                real_goal = re.findall(r'[\d]+', test_plan)[-2]
                test_plan = self.testing_data + "/" + test_plan

                os.popen("java -cp %s Recognizer -w %s %s %s %s %s %s %s %s %s" % 
                        (recognizerJar, self.models, test_plan, real_goal, obs_percentage, 50, 2.5, 2.1, 0.8, self.output_stats)).read()

                self.case_index += 1
                print(self.case_index)
                self.recentNCases()

                if self.case_index == self.totalCases:
                    break

            data = pd.read_csv("./%s" % self.output_stats, usecols=[0,1,2,3,4])
            p, r, a, bacc, std_bacc = averagedDataPoint(data, self.numModels)

            self.observation = self.observation[-self.num2relearn+1::]+[bacc[-1]]
            self.reward = 0
            self.averBACC = tailAverage(bacc, self.num2relearn)

        if self.case_index == self.totalCases:
            print("done")
            self.done = True


        info = {}

        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.observation = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
        self.reward = 0
        self.case_index = 0
        self.averBACC = 1


        os.system("rm -rf ./tmp_models")
        os.mkdir("./tmp_models")

        os.system("cp %s/*.pnml %s" % (init_models, self.models) )

        os.system("rm -rf %s" % self.output_stats)
        

        os.system("rm -rf ./Feedback")
        os.system("rm -rf %s" % self.output_stats)

        os.system("rm -rf ./*.xes")

        return self.observation  # reward, done, info can't be included



# gr_system = GRsystem(init_models, testing_data)
init_models = "/Users/zihangs/My_PHD/projects/continuous_GR/data_rl/training/driverlog_p01_hyp-1_10_1"
testing_data = "/Users/zihangs/My_PHD/projects/continuous_GR/data_rl/ori_env_output_problems_plans_drift/driverlog_p01_hyp-1_10_1"
env = CustomEnv(init_models, testing_data)


model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20)


obs = env.reset()
while True:
    print("here")
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done:
      os.system("mv %s %s" % ("mytest.csv", "out.csv") )
      obs = env.reset()
      break



