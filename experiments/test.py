import numpy as np


# [1 3 7] increasing [8 2 1] decreasing
# [1 3 3 7] increasing [1 3 2 5] False [8 2 2 1] decreasing
class Solution:
    def ArrayIsIncrease(self, array_number):
        list_array_number = np.array(array_number)
        len_list_array_number = len(list_array_number)
        
        temp_array_value_number = 0
    
        for i in range(0, len_list_array_number):
            if i > 0:
                temp_array_value_number = list_array_number[i-1]
            
            if list_array_number[i] < temp_array_value_number:
                return False
        
        return True
    
    def SecondArrayIsIncrease(self, array_number):
        list_array_number = np.array(array_number)
        len_list_array_number = len(list_array_number)
        identify_array = np.zeros(np.shape(list_array_number))
        
        temp_array_value_number = 0
        
        for i in range(0, len_list_array_number):
            if list_array_number[i] >= temp_array_value_number:
                identify_array[i] = 1
            temp_array_value_number = list_array_number[i]
            
        for i in range(0, len(identify_array)):
            if identify_array[i] == 0:
                return False
            
        return True