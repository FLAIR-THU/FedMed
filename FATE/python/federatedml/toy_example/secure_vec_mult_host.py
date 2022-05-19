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


class SecureVecMultHost(ModelBase):
    def __init__(self):
        super(SecureVecMultHost, self).__init__()
        self.y = None
        self.pty = None
        # self.y2 = None
        self.encX = None
        self.encZ = None
        # self.x2_plus_y2 = None
        self.transfer_inst = SecureVecMultExampleTransferVariable()
        self.model_param = SecureVecMultExampleParam()
        self.data_output = None
        self.model_output = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.partition = self.model_param.partition
        self.seed = self.model_param.seed

    def _init_data(self):
        a = 7
        n = self.data_num
        data2 = np.array([i * a % n + 1 for i in range(n)])
        self.y = session.parallelize(data2, include_key=False, partition=self.partition)
        LOGGER.info(list(self.y.collect()))

    # def share(self, y):
    #     first = np.random.uniform(y, -y)
    #     return first, y - first

    # def secure(self):
    #     y_shares = self.y.mapValues(self.share)
    #     self.y1 = y_shares.mapValues(lambda shares: shares[0])
    #     self.y2 = y_shares.mapValues(lambda shares: shares[1])

    def add(self):
        self.pty = PaillierTensor(self.y, partitions=self.partition)

        LOGGER.info(f"pty type is {type(self.pty)}")
        LOGGER.info(f"pty table is {type(self.pty._obj._table)}")
        LOGGER.info(f"encX type is {type(self.encX)}")
        LOGGER.info(f"encX table is {type(self.encX._obj._table)}")
        self.encZ = self.encX.multiply(self.pty)
        self.encZ = self.encZ.reduce_sum()
        # self.x2_plus_y2 = self.y2.join(self.x2, lambda y, x: y + x)
        # host_sum = self.x2_plus_y2.reduce(lambda x, y: x + y)

    def sync_share_to_guest(self):
        self.transfer_inst.host_share.remote(self.encZ,
                                             role="guest",
                                             idx=0)

    def recv_share_from_guest(self):
        self.encX = self.transfer_inst.guest_share.get(idx=0)
        self.encX = PaillierTensor(self.encX, partitions=self.partition)

    # def sync_host_sum_to_guest(self, host_sum):
    #     self.transfer_inst.host_sum.remote(host_sum,
    #                                        role="guest",
    #                                        idx=0)

    def run(self, cpn_input):
        LOGGER.info("begin to init parameters of secure add example host")
        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make host data")
        self._init_data()

        # LOGGER.info("split data into two random parts")
        # self.secure()

        LOGGER.info("get share of one random part data from guest")
        self.recv_share_from_guest()

        LOGGER.info("begin to get sum of host and guest")
        self.add()

        LOGGER.info("share one random part data to guest")
        self.sync_share_to_guest()


        # LOGGER.info("send host sum to guest")
        # self.sync_host_sum_to_guest(host_sum)

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
