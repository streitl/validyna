filenames = \
['2sp_100ex_0.01var_TEST.npz',
'2sp_100ex_1var_TRAIN.npz',
'2sp_10ex_1var_TEST.npz',
'3sp_100ex_10var_TRAIN.npz',
'3sp_10ex_10var_TEST.npz',
'2sp_100ex_0.01var_TRAIN.npz',
'2sp_10ex_0.01var_TEST.npz',
'2sp_10ex_1var_TRAIN.npz',
'3sp_100ex_1var_TEST.npz',
'3sp_10ex_10var_TRAIN.npz',
'2sp_100ex_0.1var_TEST.npz',
'2sp_10ex_0.01var_TRAIN.npz',
'3sp_100ex_0.01var_TEST.npz',
'3sp_100ex_1var_TRAIN.npz',
'3sp_10ex_1var_TEST.npz',
'2sp_100ex_0.1var_TRAIN.npz',
'2sp_10ex_0.1var_TEST.npz',
'3sp_100ex_0.01var_TRAIN.npz',
'3sp_10ex_0.01var_TEST.npz',
'3sp_10ex_1var_TRAIN.npz',
'2sp_100ex_10var_TEST.npz',
'2sp_10ex_0.1var_TRAIN.npz',
'3sp_100ex_0.1var_TEST.npz',
'3sp_10ex_0.01var_TRAIN.npz',
'2sp_100ex_10var_TRAIN.npz',
'2sp_10ex_10var_TEST.npz',
'3sp_100ex_0.1var_TRAIN.npz',
'3sp_10ex_0.1var_TEST.npz',
'2sp_100ex_1var_TEST.npz',
'2sp_10ex_10var_TRAIN.npz',
'3sp_100ex_10var_TEST.npz',
'3sp_10ex_0.1var_TRAIN.npz']

train_paths = []
test_paths = []
for f in filenames:
	if 'TEST' in f:
		test_paths.append('data/'+f)
	else:
		train_paths.append('data/'+f)
# print (train_paths)
# print (test_paths)
for trainpath in train_paths:
	for testpath in test_paths:
		os.system('python experiment.py --train_data_path='+trainpath
									+ ' --test_data_path=' +testpath 
									+ ' --autoplot')
		