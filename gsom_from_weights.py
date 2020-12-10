import sys

#sys.path.append('../../')

import sys
for path in sys.path:
    print(path)

sys.path.append('/root/ThEmoBe/Parallel_GSOM_for_HAAP')
sys.path.append('/root/ThEmoBe/Parallel_GSOM_for_HAAP')

print("Adding core4")
for path in sys.path:
    print(path)

import numpy as np
import copy
import threading

from core4 import growth_handler as Growth_Handler
from util import utilities as Utils


np.random.seed(8)


class GSOM_from_Weights():

    def __init__(self, params):
        threading.Thread.__init__(self)
        self.parameters = params
        self.growth_handler = Growth_Handler.GrowthHandler()
        self.gsom_nodemap = {}

        # Parameters for recurrent gsom
        self.alphas = Utils.Utilities.get_decremental_alphas(self.parameters.NUMBER_OF_TEMPORAL_CONTEXTS)

    def loadWeights(self, nodemap):
        self.gsom_nodemap = nodemap

    def predict(self, X_test):
        y_pred = []
        param = self.parameters
        gsom_nodemap = copy.deepcopy(self.gsom_nodemap)

        for cur_input in X_test:
            self.globalContexts_evaluation[0] = cur_input

            # Update global context
            for z in range(1, param.NUMBER_OF_TEMPORAL_CONTEXTS):
                self.globalContexts_evaluation[z] = (param.BETA * self.previousBMU_evaluation[0, z]) + (
                        (1 - param.BETA) * self.previousBMU_evaluation[0, z - 1])

            winner = Utils.Utilities.select_winner_recurrent(gsom_nodemap, self.globalContexts_evaluation, self.alphas)
            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            y_pred.append(winner.get_mapped_labels())
        return y_pred

    # return predictions and weights
    def predict_x(self, X_test):
        y_pred = []
        weights = []

        gsom_nodemap = copy.deepcopy(self.gsom_nodemap)
        for cur_input in X_test:
            winner = Utils.Utilities.select_winner(gsom_nodemap, np.array([cur_input]))
            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            y_pred.append(winner.get_mapped_labels())
            weights.append(winner.recurrent_weights)

        return y_pred, weights  