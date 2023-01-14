import os
import re
import subprocess
import numpy as np
import pandas as pd
import statistics
import shutil

from sklearn import linear_model

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

def reCreateDir(dirName):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dirName)
    if isExist:
        # delete
        shutil.rmtree(dirName)
    
    os.makedirs(dirName)



############################# systems ################################
class GRsystem:
    def __init__(self, init_models, testing_data):
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

    def countModels(self):
        models = 0
        for item in os.listdir(self.models):
            if item[-5::] == ".pnml":
                models += 1
        return models

    def recognizer(self, test_plan, real_goal, obs_percentage, output_stats, phi = 50, lamb=2.5, delta=2.1, threshold=0.8):
        os.popen("java -cp %s Recognizer -w %s %s %s %s %s %s %s %s %s" % 
                (self.recognizerJar, self.models, test_plan, real_goal, obs_percentage, phi, lamb, delta, threshold, output_stats)).read()

    def controllerOpenLoop(self, option):
        os.system("java -jar %s %s %s %s" % (self.controllerJar, option, self.relearn_dir, self.num2relearn) )
        
    def controllerClosedLoopAve(self, option, current_acc, acc_threshold):
        os.system("java -jar %s %s %s %s %s" % (self.controllerJar, option, self.relearn_dir, current_acc, acc_threshold) )
    
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


    def run(self, obs_percentage, output_stats, option):
        #: options: "-no-relearn"
        ## delete (prev) feedbacks and (prev) csv file
        os.system("rm -rf ./Feedback")
        os.system("rm -rf %s" % output_stats)

        case_index = 0
        remine_count = 0

        # only for closed loop
        highest_n_acc = 0

        for test_plan in self.test_list:
            if test_plan == ".DS_Store":
                continue
            else:
                case_index += 1
                
            # real_goal = test_plan.split(".")[0].split("_")[1] + ".pnml"
            real_goal = re.findall(r'[\d]+', test_plan)[-2]
            test_plan = self.testing_data + "/" + test_plan
            print(test_plan)

            # recognize a single GR task
            self.recognizer(test_plan, real_goal, obs_percentage, output_stats)

            # check if to relearn
            if option == "-no-relearn":
                pass

            if option == "-openloop": # -openloop: consists with jar file
                # need call the controller
                self.controllerOpenLoop(option)
                # remine model or not?
                relearnFlag = self.checkIfRemine()
                # add flag
                self.addRelearnFlag(output_stats, relearnFlag)

            if option == "-closedloop_ave_metric":
                ## prepare data to relearn: collect X recent cases for all goals
                self.recentNCases()
                # check statistics
                # usecols= ['Real_Goal','Time','Cost','Prob','Results','Relearn']
                data = pd.read_csv("./%s" % output_stats, usecols=[0,1,2,3,4])
                # p, r, a, bacc = calculate_statistics(data)
                p, r, a, bacc, std_bacc = averagedDataPoint(data, self.numModels)

                # calculate average bacc for every period and keep highest.
                if case_index >= self.num2relearn*self.numModels:
                    recent_n_acc = tailAverage(bacc, self.num2relearn)
                else:
                    recent_n_acc = 0

                if recent_n_acc > highest_n_acc:
                    highest_n_acc = recent_n_acc

                # if check to relearn:
                if case_index%(self.num2relearn*self.numModels) == 0 and len(bacc) >= self.num2relearn:
                    
                    # recent_n_acc = tailAverage(bacc, self.num2relearn)

                    # set threshold for relearn
                    # acc_threshold = tailAverage(bacc[-2*self.num2relearn : -self.num2relearn], self.num2relearn) * 0.8
                    acc_threshold = highest_n_acc * 0.8
                    self.controllerClosedLoopAve(option, recent_n_acc, acc_threshold)

                # remine model or not?
                relearnFlag = self.checkIfRemine()
                # add flag
                self.addRelearnFlag(output_stats, relearnFlag)

            if option == "-closedloop-trend":
                self.recentNCases()
                data = pd.read_csv("./%s" % output_stats, usecols=[0,1,2,3,4])
                p, r, a, bacc, std_bacc = averagedDataPoint(data, self.numModels)

                # calculate average bacc for every period and keep highest.
                if case_index >= self.num2relearn*self.numModels:
                    recent_n_acc = tailAverage(bacc, self.num2relearn)
                else:
                    recent_n_acc = 0

                if recent_n_acc > highest_n_acc:
                    highest_n_acc = recent_n_acc

                # if check to relearn:
                if case_index%(self.num2relearn*self.numModels) == 0 and len(bacc) >= self.num2relearn:
                    # next n acc
                    pointID = len(bacc)
                    X = np.array(range(pointID-self.num2relearn+1, pointID+1, 1)).reshape(-1,1)
                    reg = linear_model.LinearRegression()
                    reg.fit(X, bacc[-self.num2relearn::])
                    next_n_acc = reg.predict( np.array(pointID+self.num2relearn).reshape(-1,1) )

                    # set threshold for relearn
                    # acc_threshold = tailAverage(bacc[-2*self.num2relearn : -self.num2relearn], self.num2relearn) * 0.8
                    acc_threshold = highest_n_acc * 0.8


                    print(next_n_acc)
                    print(acc_threshold)

                    self.controllerClosedLoopAve("-closedloop_ave_metric", next_n_acc[0], acc_threshold)

                # remine model or not?
                relearnFlag = self.checkIfRemine()
                # add flag
                self.addRelearnFlag(output_stats, relearnFlag)



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




if __name__ == "__main__":

    ################################ parameters ###################################
    test_dataset = "experimental_data/data_gradual/ori_env_output_problems_plans_drift"
    training_dataset = "experimental_data/data_gradual/training"

    obs_percentage = 0.5

    # system options: 1. "-no-relearn"; 
    #                 2. "-openloop"; 
    #                 3. "-closedloop_ave_metric"; 
    #                 4. "-closedloop-trend";
    option = "-openloop"

    output_dir = "results"
    ###############################################################################


    reCreateDir(output_dir)
    tests = os.listdir(test_dataset)
    for domain in tests:
        if domain == ".DS_Store":
            continue
        for problem_name in os.listdir(os.path.join(test_dataset, domain)):
            if problem_name == ".DS_Store":
                continue

            init_models = os.path.join(training_dataset, domain, problem_name) # initial models
            testing_data = os.path.join(test_dataset, domain, problem_name)

            output_stats = os.path.join(output_dir, "%s_%s_%s_per%s.csv" % (domain, problem_name, option, str(obs_percentage*100) ) )

            gr_system = GRsystem(init_models, testing_data)
            gr_system.run(obs_percentage, output_stats, option)







