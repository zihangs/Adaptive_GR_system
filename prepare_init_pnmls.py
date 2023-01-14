import os

# Mine PNML from set of plans
############################ iterative ############################

def iter_convert(path):
	# check if it is set of plans:
	is_set_of_plans = False
	lstdir = os.listdir(path)
	for subPath in lstdir:
		if subPath[0] != "." and os.path.isfile(path + "/" + subPath) and (path + "/" + subPath).split("/")[-1][0:3] == "sas":
			is_set_of_plans = True
			break

	if is_set_of_plans:
		print("This is a set of plan: " + path)
		# convert plans to event logs:
		os.system("java -jar sas2xes.jar %s %s.xes" % (path, path))
		
		# java -jar sas2xes.jar found_plans out.xes

		return path

	else:
		for subPath in lstdir:
			if subPath[0] != ".":
				iter_convert(path + "/" + subPath)


##############################################################
if __name__ == "__main__":
	training_data = "data/training"

	iter_convert(training_data)

	os.system("mv %s ./miningPNMLS/" % training_data)
	os.chdir("./miningPNMLS")
	os.system("java -jar mine_all_pnmls.jar -DFM ./training/ 0.8")
	os.system("mv training ../%s" % training_data)
	os.chdir("../")




