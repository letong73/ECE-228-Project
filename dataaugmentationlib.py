import math

import numpy as np

THETA_1 = math.pi/2
THETA_2 = math.pi
THETA_3 = (3*math.pi)/2

SIGMA_1 = 0.0005
SIGMA_2 = 0.001
SIGMA_3 = 0.002


def rotate(signals, labels):
    """
        This function creates a list containing three rotated copies of each element in signals. Rotations are done by
        90°, 180° and 270°.

        Args:
            signals: numpy.array of 2x128 matrixes each one representing a signal.
            labels: labels for each signal.

        Returns:
            numpy.array of 2x128 matrixes each one representing a rotated signal.
    """

    # Rotated signal matrix B is obtained by multiplication of T transformation matrix with A original signal matrix.
    #
    # B = TxA
    #
    # T is defined as shown below.
    #
    # | cos(theta)  -sin(theta) |
    # | sin(theta)   cos(theta) |

    def T(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    T_1 = T(THETA_1)
    T_2 = T(THETA_2)
    T_3 = T(THETA_3)

    # list containing rotated signals
    rotated_signals = []
    rotated_labels = []

    # for each signal B = TxA
    for i in range(0, len(signals)):
        A = signals[i]
        l = labels[i]

        # rotate by THETA_1
        B = np.matmul(T_1, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

        # rotate by THETA_2
        B = np.matmul(T_2, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

        # rotate by THETA_3
        B = np.matmul(T_3, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

    return rotated_signals, rotated_labels


def rotate_and_concatenate_with_signals(signals, labels):
    """
        This function creates a list containing three rotated copies of each element in signals, concatenated with the
        given signals list. Rotations are done by 90°, 180° and 270°.

        Args:
            signals: numpy.array of 2x128 matrixes each one representing a signal.
            labels: labels for each signal.

        Returns:
            numpy.array of 2x128 matrixes each one representing a rotated signal.
    """

    rotated_signals, rotated_labels = rotate(signals, labels)

    return np.concatenate((signals, rotated_signals)), np.concatenate((labels, rotated_labels))



