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
import sys
print(sys.path)
sys.path.append('/data/projects/fate/fate/python')
sys.path.append('/data/projects/fate/fateflow/python')
import argparse
import os

from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils.tools import load_job_config
from tensorflow.keras import layers

def main(config="../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {"name": "zs_c2_guest_org", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "zs_c2_host_org", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=50,
                           interactive_layer_lr=0.15, batch_size=100, early_stop="diff")
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    # guest_nn_0.add_bottom_model(Dense(units=7, input_shape=(9,), activation="relu",
    #                                   kernel_initializer=initializers.Constant(value=1)))
    # guest_nn_0.add_bottom_model(Dense(units=5, input_shape=(7,), activation="relu",
    #                                   kernel_initializer=initializers.Constant(value=1)))
    #                                   kernel_initializer=initializers.Constant(value=1)))
    guest_nn_0.add_bottom_model(Dense(units=9, input_shape=(9,),use_bias=True,bias_initializer=initializers.Constant(value=-0.5)))
    guest_nn_0.add_bottom_model(Dense(units=7, input_shape=(9,), activation="relu",use_bias=True,bias_initializer=initializers.Zeros()))
    guest_nn_0.add_bottom_model(Dense(units=5, input_shape=(7,), activation="relu",use_bias=True,bias_initializer=initializers.glorot_uniform(seed=None)))
    guest_nn_0.set_interactve_layer(Dense(units=4, input_shape=(5,)))
    guest_nn_0.add_top_model(Dense(units=2, input_shape=(4,), activation="relu"))
    # guest_nn_0.add_top_model(Dense(units=1, input_shape=(2,)))
    guest_nn_0.add_top_model(Dense(units=1, input_shape=(1,), use_bias=True, activation="sigmoid"))
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)

    input_shape = (2700,)
    host_nn_0.add_bottom_model(layers.Reshape((30, 30, 3), input_shape=input_shape))
    host_nn_0.add_bottom_model(layers.Conv2D(8, kernel_size=(3, 3), activation="relu"))
    host_nn_0.add_bottom_model(layers.MaxPooling2D(pool_size=(2, 2)))
    host_nn_0.add_bottom_model(layers.Conv2D(16, kernel_size=(3, 3), activation="relu"))
    host_nn_0.add_bottom_model(layers.MaxPooling2D(pool_size=(2, 2)))
    host_nn_0.add_bottom_model(layers.Flatten())
    host_nn_0.add_bottom_model(Dense(units=8))
    host_nn_0.add_bottom_model(Dense(units=8, activation="relu"))
    # host_nn_0.add_bottom_model(Dense(units=8, input_shape=(100,), activation="relu",
    #                                  kernel_initializer=initializers.Constant(value=1)))
    host_nn_0.set_interactve_layer(Dense(units=4, input_shape=(8,),
                                         kernel_initializer=initializers.Constant(value=1)))
    # hetero_nn_0.compile(optimizer=optimizers.SGD(lr=0.015), loss="binary_crossentropy")
    hetero_nn_0.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")

    hetero_nn_1 = HeteroNN(name="hetero_nn_1")

    evaluation_0 = Evaluation(name="evaluation_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_nn_1, data=Data(test_data=intersection_0.output.data),
                           model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    pipeline.compile()

    pipeline.fit()

    print(hetero_nn_0.get_config(roles={"guest": [guest],
                                        "host": [host]}))
    print(pipeline.get_component("hetero_nn_0").get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
