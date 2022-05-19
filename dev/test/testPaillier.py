import sys
print(sys.path)
# sys.path.append('/data/projects/fate/fate/python')
# sys.path.append('/data/projects/fate/fateflow/python')
sys.path.append('/home/cjp/FATE/python')
from fate_arch.common import file_utils
file_utils.PROJECT_BASE ="/data/projects/fate"
print(file_utils.get_project_base_directory())
# 重要前缀内容

from fate_arch.session import computing_session
import numpy as np
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.secureprotol import EncryptModeCalculator

def myGetNumpy(table):
    arr1 = [None for i in range(table.count())]
    for k, v in table.collect():
        arr1[k] = v
    res = np.array(arr1)
    return res
def myGetNumpyFromPaillierTensor(ts):
    return myGetNumpy(ts._obj)

computing_session.init(session_id="a great session")
key_length=128
cipher = PaillierEncrypt()
cipher.generate_key(key_length)
pub_key = cipher.get_public_key()
a=cipher.encrypt(3)
b=cipher.encrypt(8)
c=a*3.14
print(cipher.decrypt(c))
encrypted_calculator = EncryptModeCalculator(cipher,"fast")
n=13;
a=7;
data1 = np.cos(1+np.array(list(range(20))))
data2 = np.array([i*a%n+1 for i in range(n)])

data2 = computing_session.parallelize(data2, include_key=False, partition=10)
data1=computing_session.parallelize(data1, include_key=False, partition=2)
paillier_tensor1 = PaillierTensor(data1, partitions=10)
paillier_tensor2 = PaillierTensor(data2, partitions=10)
encPt1=paillier_tensor1.encrypt(encrypted_calculator)
encPt2=paillier_tensor2.encrypt(encrypted_calculator)
encPt3=encPt1.multiply(paillier_tensor2)
encPt4=encPt3.reduce_sum()
arr = myGetNumpyFromPaillierTensor(encPt3)
print(arr)
dePt3=encPt3.decrypt(cipher)
arr = myGetNumpyFromPaillierTensor(dePt3)
print(arr)
v=cipher.decrypt(encPt4)
print(v)


# arr = paillier_tensor1.numpy()
# print(arr)
# encPt1=paillier_tensor1.encrypt(encrypted_calculator)
# arr = myGetNumpyFromPaillierTensor(encPt1)
# print(arr)
# dePt1=encPt1.decrypt(cipher)
# arr = myGetNumpyFromPaillierTensor(dePt1)
# print(arr)
# paillier_tensor1+paillier_tensor2