import pandas as pd
import numpy as np
import re
import MergeService
import csv

def answer_one():

    dataFrame = MergeService.getMergedFrames()

    return dataFrame

print(answer_one())
