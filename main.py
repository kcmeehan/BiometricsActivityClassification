#!/usr/bin/python

from functions import preProcess

if __name__ == '__main__':

  # Make this a for loop over all data files eventually, 
  # but for now just test with one file
  df101 = preProcess('../PAMAP2_Dataset/Protocol/subject101.dat') 
