import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

model = tf.keras.models.load_model(
    "best_model.keras",
    compile=False
)

def generate_points(image):
    # Generate the 2D mask and initialise variables
    prediction = model.predict(image)
    mask = (prediction["mask"].squeeze() > 0.95).astype(int)
    points_list = []
    two_branches = False

    # Find the center of every branch horizontally
    for row in range(255):
        start = "NaN"
        new_points = []

        for column in range(255):
            # Check for white pixel, if white wait until black then find the middle.
            if (mask[row, column] == 1) & (start == "NaN"):
                start = column
            elif (mask[row, column] == 0) & (start != "NaN"):
                middle = (start + column) / 2
                radius = (column - start) / 2 
                new_points.append((middle, row, 0, radius))
                start = "NaN"
        if (len(new_points) == 2):
            two_branches = True

        if (two_branches == True) & (len(new_points) == 1):
            end_double_row = row
            final_double_row = (points_list[-2], points_list[-1])
            break
        points_list = points_list + new_points

    radius_multiplier = 1

    for row in range(255, end_double_row, -1):
        start = "NaN"
        new_points = []

        for column in range(255):
            # Check for white pixel, if white wait until black then find the middle.
            if (mask[row, column] == 1) & (start == "NaN"):
                start = column

            elif (mask[row, column] == 0) & (start != "NaN"):
                middle = (start + column) / 2
                radius = (column - start) / 2 * radius_multiplier
                new_points.append((middle, row, 0, radius))
                start = "NaN"

        points_list = points_list + new_points

        if (row - radius < end_double_row):
            final_single_row = points_list[-1]
            break

    # Linear interpolation between the ends of the single and ends of the double branches.
    num_interpolations = 4
    for branch in final_double_row:
        interpolations = np.linspace(branch, final_single_row, num_interpolations + 2)[1:-1]
        points_list.extend(map(tuple, interpolations.tolist()))

    return points_list