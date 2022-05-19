#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.model_base import ModelBase, ComponentOutput
# from federatedml.param.secure_add_example_param import SecureAddExampleParam
from federatedml.transfer_variable.transfer_class.secure_vec_mult_example_transfer_variable import \
    SecureVecMultExampleTransferVariable
from federatedml.util import LOGGER
from federatedml.param.secure_vec_mult_example_param import SecureVecMultExampleParam
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol import EncryptModeCalculator


class SecureVecMultGuest(ModelBase):
    def __init__(self):
        super(SecureVecMultGuest, self).__init__()
        self.x = None
        # self.x1 = None
        # self.x2 = None
        self.ptx = None
        # self.x1_plus_y1 = None
        self.z = None
        self.data_num = None
        self.partition = None
        self.seed = None
        self.transfer_inst = SecureVecMultExampleTransferVariable()
        self.model_param = SecureVecMultExampleParam()
        self.data_output = None
        self.model_output = None
        self.cipher = None
        self.pub_key = None
        self.encrypted_calculator = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.partition = self.model_param.partition
        self.seed = self.model_param.seed
        key_length = 128
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(key_length)
        self.pub_key = self.cipher.get_public_key()
        self.encrypted_calculator = EncryptModeCalculator(self.cipher, "fast")

    def _init_data(self):
        kvs = np.cos(1+np.array(list(range(self.data_num))))
        self.x = session.parallelize(kvs, include_key=False, partition=self.partition)
        LOGGER.info(list(self.x.collect()))


    # def share(self, x):
    #     first = np.random.uniform(x, -x)
    #     return first, x - first

    def secure(self):
        self.ptx = PaillierTensor(self.x, partitions=self.partition)
        self.ptx = self.ptx.encrypt(self.encrypted_calculator)


    # def add(self):
    #     self.x1_plus_y1 = self.x1.join(self.y1, lambda x, y: x + y)
    #     guest_sum = self.x1_plus_y1.reduce(lambda x, y: x + y)
    #     return guest_sum

    def reconstruct(self):
        # LOGGER.info("host sum is %.4f" % host_sum)
        LOGGER.info("received Enc Res is %s" % self.z.__str__())
        secure_res = self.cipher.decrypt(self.z)

        LOGGER.info("Dec Secure Result is %.4f" % secure_res)

        return secure_res

    def sync_share_to_host(self):
        LOGGER.info(f"encX type is {type(self.ptx)}")
        # LOGGER.info(f"encX table is {type(self.ptx._obj._table)}")
        self.transfer_inst.guest_share.remote(self.ptx._obj._table,
                                              role="host",
                                              idx=0)

    def recv_share_from_host(self):
        self.z = self.transfer_inst.host_share.get(idx=0)

    # def recv_host_sum_from_host(self):
    #     host_sum = self.transfer_inst.host_sum.get(idx=0)
    #
    #     return host_sum

    def run(self, cpn_input):
        LOGGER.info("begin to init parameters of secure add example guest")

        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make guest data")
        self._init_data()

        LOGGER.info("split data into two random parts")
        self.secure()

        LOGGER.info("share one random part data to host")
        self.sync_share_to_host()

        LOGGER.info("get share of one random part data from host")
        self.recv_share_from_host()

        # LOGGER.info("begin to get sum of guest and host")
        # guest_sum = self.add()
        #
        # LOGGER.info("receive host sum from guest")
        # host_sum = self.recv_host_sum_from_host()

        secure_sum = self.reconstruct()

        # assert (np.abs(secure_sum - self.data_num * 2) < 1e-6)

        LOGGER.info("success to calculate secure_sum, it is {}".format(secure_sum))

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
