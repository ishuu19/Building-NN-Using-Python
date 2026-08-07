import numpy as np


# "W←W−ηdW"
# "b←b−ηdb"

class SGD:
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "weights"):
                self.weights -= self.learning_rate * layer.dweights
                self.biases -= self.learning_rate * layer.dbiases



class Momentum:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.learning_rate = learning_rate
        self.beta = beta

        self.velocities = {}


    def step(self, layers):

        for layer in layers:

            if not hasattr(layer, "weights"):
                continue


            if id(layer) not in self.velocities:

                self.velocities[id(layer)] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }


            velocity = self.velocities[id(layer)]


            velocity["weights"] = (
                self.beta * velocity["weights"]
                + layer.dweights
            )

            velocity["biases"] = (
                self.beta * velocity["biases"]
                + layer.dbiases
            )


            layer.weights -= (
                self.learning_rate
                * velocity["weights"]
            )

            layer.biases -= (
                self.learning_rate
                * velocity["biases"]
            )


class Nesterov:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.learning_rate = learning_rate
        self.beta = beta

        self.velocities = {}


    def step(self, layers):

        for layer in layers:

            if not hasattr(layer, "weights"):
                continue


            if id(layer) not in self.velocities:

                self.velocities[id(layer)] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }


            velocity = self.velocities[id(layer)]


            old_vw = velocity["weights"].copy()
            old_vb = velocity["biases"].copy()


            velocity["weights"] = (
                self.beta * velocity["weights"]
                - self.learning_rate * layer.dweights
            )

            velocity["biases"] = (
                self.beta * velocity["biases"]
                - self.learning_rate * layer.dbiases
            )


            layer.weights += (
                -self.beta * old_vw
                + (1 + self.beta) * velocity["weights"]
            )

            layer.biases += (
                -self.beta * old_vb
                + (1 + self.beta) * velocity["biases"]
            )



class RMSProp:

    def __init__(
        self,
        learning_rate=0.001,
        beta=0.9,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

        self.cache = {}


    def step(self, layers):

        for layer in layers:

            if not hasattr(layer, "weights"):
                continue


            if id(layer) not in self.cache:

                self.cache[id(layer)] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }


            cache = self.cache[id(layer)]


            cache["weights"] = (
                self.beta * cache["weights"]
                + (1 - self.beta) * layer.dweights ** 2
            )

            cache["biases"] = (
                self.beta * cache["biases"]
                + (1 - self.beta) * layer.dbiases ** 2
            )


            layer.weights -= (
                self.learning_rate
                * layer.dweights
                / (
                    np.sqrt(cache["weights"])
                    + self.epsilon
                )
            )


            layer.biases -= (
                self.learning_rate
                * layer.dbiases
                / (
                    np.sqrt(cache["biases"])
                    + self.epsilon
                )
            )



class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.t = 0

        self.m = {}
        self.v = {}


    def step(self, layers):

        self.t += 1


        for layer in layers:

            if not hasattr(layer, "weights"):
                continue


            key = id(layer)


            if key not in self.m:

                self.m[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }

                self.v[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }


            # -------------------------
            # FIRST MOMENT
            # -------------------------

            self.m[key]["weights"] = (
                self.beta1 * self.m[key]["weights"]
                + (1 - self.beta1) * layer.dweights
            )

            self.m[key]["biases"] = (
                self.beta1 * self.m[key]["biases"]
                + (1 - self.beta1) * layer.dbiases
            )


            # -------------------------
            # SECOND MOMENT
            # -------------------------

            self.v[key]["weights"] = (
                self.beta2 * self.v[key]["weights"]
                + (1 - self.beta2) * layer.dweights ** 2
            )

            self.v[key]["biases"] = (
                self.beta2 * self.v[key]["biases"]
                + (1 - self.beta2) * layer.dbiases ** 2
            )


            # -------------------------
            # BIAS CORRECTION
            # -------------------------

            m_weights_corrected = (
                self.m[key]["weights"]
                / (1 - self.beta1 ** self.t)
            )

            m_biases_corrected = (
                self.m[key]["biases"]
                / (1 - self.beta1 ** self.t)
            )


            v_weights_corrected = (
                self.v[key]["weights"]
                / (1 - self.beta2 ** self.t)
            )

            v_biases_corrected = (
                self.v[key]["biases"]
                / (1 - self.beta2 ** self.t)
            )


            # -------------------------
            # UPDATE PARAMETERS
            # -------------------------

            layer.weights -= (
                self.learning_rate
                * m_weights_corrected
                / (
                    np.sqrt(v_weights_corrected)
                    + self.epsilon
                )
            )


            layer.biases -= (
                self.learning_rate
                * m_biases_corrected
                / (
                    np.sqrt(v_biases_corrected)
                    + self.epsilon
                )
            )



            







            












            
        
        