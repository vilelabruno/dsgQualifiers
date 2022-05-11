import pip
c = 0
while c < 2:
        with open('pipReq.log', 'a') as the_file:
                with open('r3.txt', 'r') as fpW:
                        lines = fpW.readlines()
                        with open('r5.txt', 'w') as fp:
                                cnt = 1
                                for line in lines:
                                        print("Line {}: {}".format(cnt, line))
                                        response = pip.main(['install', line]) #0 to success 1 to fail
                                        if response == 1:
                                                fp.write(line)
                                        the_file.write('Package {} status {}\n'.format(line, response))
                                        cnt += 1
        c = c+1













