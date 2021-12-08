import numpy as np

class Average:
    def Average(lst):
        return sum(lst) / len(lst)
        
    def NoneAverage(lst):
        out_arr = np.zeros(len(lst[0]), dtype=int)
        for item in lst:
            out_arr = np.add(out_arr, item)  
        results = np.divide(out_arr, len(lst))
        return results

    def cmAverage(lst):
        s = (len(lst[0]),4)
        out_arr = np.zeros(s, dtype=int)
        for item in lst:
            out_arr = np.add(out_arr, item)  
        results = np.divide(out_arr, len(lst))
        return results 
    
if __name__=='__main__': 
    print('DONE')