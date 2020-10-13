# runBCT

runBCT is a wrapper script to make the bct toolbox more functional. 

In order to run runBCT, you must have the bctnet toolbox downloaded and in your directory (https://sites.google.com/site/bctnet)

There are two arguments to runBCT, corrmatrix is the connectivity matrix in the form of a npy array [roi x roi x session].
The sessionfile is a csv file with information reagarding each scan (subject ID, timepoint, group etc)

EXAMPLE:

rb = RunBCT('test_data.npy', 'session_info.csv')

rb.load_files()

den =  rb.get_density(rb.corr_matrix)
