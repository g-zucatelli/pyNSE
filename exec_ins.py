import time
from pyINS_class import INS

if __name__ == '__main__':
    ins = INS("temp_list.txt", 16000, [0.003, 0.008, 0.012, 0.016, 0.25, 0.3], 5)
    ins.exec_ins_calc()

    