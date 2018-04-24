import numpy as np
import os
import shutil

percentages = [0.8, 0.1, 0.1]
datadir = "./data/"
rootdir = "." # change to '/data' if used in docker image

np.random.seed(1314)

classes = {}
for idx, fold in enumerate(sorted(os.listdir(datadir))):
	classes[fold] = idx
	os.makedirs("./train/"+fold)
	os.makedirs("./test/"+fold)
	os.makedirs("./validation/"+fold)

	# remove all png files and then
	# print the directory for the current directory
	currdir = datadir + fold
	filelist = [f for f in os.listdir(currdir) if f.endswith(".png")]
	for f in filelist:
		os.remove(os.path.join(currdir, f))
	files = sorted(os.listdir(currdir))
	print("There are [{}] files in {}.\nThe first 5 files are {}".format(len(files),
																	   currdir,
																	   files[:5]))

	# shuffle the filelist and print the first 5 files
	np.random.shuffle(files)
	print("After shuffle, the first 5 files are: {}\n".format(files[:5]))

	# prepare the split percentage
	split1 = int(len(files)*(percentages[0]))
	split2 = int(len(files)*(1-percentages[-1]))

	# copy files into separate subfolders
	for idx, f in enumerate(files):
		if idx < split1:
			shutil.copy(os.path.join(currdir, f), os.path.join("./train/"+fold, f))
		elif idx < split2:
			shutil.copy(os.path.join(currdir, f), os.path.join("./validation/"+fold, f))
		else:
			shutil.copy(os.path.join(currdir, f), os.path.join("./test/"+fold, f))

print("[INFO] Data preparation success. ") 

# prepare test csv file
print("Classes index mapping: {}".format(classes))
with open("test.csv", "w") as _file:
	for root, dirs, fns in os.walk("./test"):
		for fn in fns:
			# fpath = os.path.join(root, fn)
			for k in sorted(classes.keys()):
				if k in root:
					_file.write("{},{}\n".format(os.path.join(rootdir+root[1:], fn), classes[k]))
				continue
print("[INFO] Test file list preparation success. ")
