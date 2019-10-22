# 20191021
### Asssignment 01

#  Geooprocessing with Python
#  Vincent Hanuschik

#  Exercise I – Function (1)
#      - Write a function, that calculates the square of a number.
#      Apply the function to a list of numbers using the map() function
#      and print the result to the screen. Go through the provided books,
#      or have a look online on how the map()-functions works.
#
#  Exercise II – Function (2)
#       - Write a function, that calculates the square root of a number
#       (note: the sqrt-function of the math- package does exactly this).
#       The function we want you to write should accept a single number as argument.
#       As you know, for some numbers it is not possible to calculate the square root,
#       such as negative values. Your function should be able to handle these numbers
#       without producing an error message – instead, it should provide in a print statement
#       on the screen indicating to the user that from negative numbers your function cannot
#       calculate the square root.
# ####

import time
import os
import math as m


### Exercise 1 ###

def square(x):
    '''
    #calculates the square of a number x
    '''
    return x ** 2


# list to test function on. Includes integers, negative numbers and floats
my_list = [4, 9, -10, 1.192, 123091283]

# list must be put before map() in order to return list
list(map(square, my_list))  # map(function, object)


### Exercise 2 ###

def my_sqrt(x):
    '''
    Calculates the square root of x using the sqrt of the math package (m. instead of math.)
    Checks x first to exclude negative numbers with if-function.
    '''
    if x < 0:
        print('Not possible to calculate square root of this number.')
        return 0
    else:
        z = m.sqrt(x)
    return z


# list must be put before map() in order to return list
list(map(my_sqrt, my_list))  # map(function, object)