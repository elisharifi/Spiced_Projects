"""
Start with this to implement the supermarket simulator.
"""

import numpy as np
import pandas as pd


class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self):        
        # a list of Customer objects
        self.customers = []
        self.minutes = 0
        self.last_id = 0

    def __repr__(self):
        return ''

    def get_time(self):
        """current time in HH:MM format,
        """
        return None

    def print_customers(self):
        """print all customers with the current time and id in CSV format.
        """
        return None

    def next_minute(self):
        """propagates all customers to the next state.
        """
        return None
    
    def add_new_customers(self):
        """randomly creates new customers.
        """
        return None

    def remove_exitsting_customers(self):
        """removes every customer that is not active any more.
        """
        return None
