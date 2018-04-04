from subprocess import run

#param1 = int(sys.argv[1])#hidden1
#param2 = int(sys.argv[2])#hidden2
#param3 = int(sys.argv[3])#batch_size

#param4 = int(sys.argv[4])#broken_sensor
    #if param4 == 1:
    #    前方3つ故障
    #elif param4 ==2:
    #    ノイズ0.9〜1.1
    #else:
    #    no_broken_sensor

#param5 = int(sys.argv[5])#memory
#param6 = int(sys.argv[6])#episodes
#param7 = int(sys.argv[7])#num_obstacles
#param8 = int(sys.argv[8])#pos_obstacles
    #if param8 == 0:
    #obs=setted     
    #else:    
    #obs=rand

#param_name = str(sys.argv[9])#filename



samples = [(128,128,32,0,50000,5000,5,1,'128128test'),
           (128,128,32,0,50000,5000,5,1,'128128test2') ]


for sample in samples:
   run('python3 dqn_obstacle.py {} {} {} {} {} {} {} {} {}'.format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8]), shell=True)
