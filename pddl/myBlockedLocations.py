import random
import sys

def generateBlockedBlocks(seed):
    mapping = {2:'loc_x1y2',3:'loc_x1y3',4:'loc_x1y4',5:'loc_x2y1',6:'loc_x2y2',7:'loc_x2y3',8:'loc_x2y4',9:'loc_x3y1',10:'loc_x3y2',11:'loc_x3y3',12:'loc_x3y4',13:'loc_x4y1',14:'loc_x4y2',15:'loc_x4y3'}
    random.seed(seed)
    block1 = random.randint(2,15)
    block2 = random.randint(2,15)
    if((block1 == 2 and block2 == 5) or (block1 == 5 and block2 == 2)):
        block1 = 7
    elif((block1 == 12 and block2 == 15) or (block1 == 15 and block2 == 12)):
        block1 = 7
    print('Your blocked locations are {} and {}'.format(mapping[block1], mapping[block2]))

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        generateBlockedBlocks(int(sys.argv[1])) 
    else:
        print('Please provide your ASU ID as an argument while running this file .i.e myBlockedLocations 123432123')
