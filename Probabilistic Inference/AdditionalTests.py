"""
Author: kachtani3
Description: Additional unit tests for 3630 Spring 2020 P2. 
"""

from Project2 import *
import unittest

class AdditionalUnitTests(unittest.TestCase):
    def assertEq(self, actual, expected, message):
        self.assertGreaterEqual(actual, expected - 1e-6, message)
        self.assertLessEqual(actual, expected + 1e-6, message)

    def test_cdf_basic(self) -> None:
        pmf = {
            'A': 0.1,
            'B': 0.2,
            'C': 0.5,
            'D': 0.2
        }

        def test_pmf(val):
            return pmf[val]

        cdf = calculate_cdf(test_pmf, list(pmf.keys()))

        self.assertEq(cdf('A'), 0.1, "Incorrect cdf for A")
        self.assertEq(cdf('B'), 0.3, "Incorrect cdf for B")
        self.assertEq(cdf('C'), 0.8, "Incorrect cdf for C")
        self.assertEq(cdf('D'), 1.0, "Incorrect cdf for D")

    def test_cdf_2_value(self) -> None:
        pmf = {
            'A': 0.0,
            'B': 1.0
        }

        def test_pmf(val):
            return pmf[val]

        cdf = calculate_cdf(test_pmf, list(pmf.keys()))

        self.assertEq(cdf('A'), 0.0, "Incorrect cdf for A")
        self.assertEq(cdf('B'), 1.0, "Incorrect cdf for B")

    def test_transition_model_edge(self) -> None:
        stay_from_top = transition_model(T=(0, 5), S=(0, 5), A='Up')
        left_from_top = transition_model(T=(0, 4), S=(0, 5), A='Up')
        right_from_top = transition_model(T=(0, 6), S=(0, 5), A='Up')
        down_from_top = transition_model(T=(1, 5), S=(0, 5), A='Up')

        self.assertEq(stay_from_top, 0.85, "Incorrect transition model when staying in cell from top row")

        for val in [left_from_top, right_from_top, down_from_top]:
            self.assertEq(val, 0.05, "Incorrect transition model when not staying in cell from top row")

    def test_transition_model_corner(self) -> None:
        stay_from_corner = transition_model(T=(0, 9), S=(0, 9), A='Right')
        down_from_corner = transition_model(T=(1, 9), S=(0, 9), A='Right')
        left_from_corner = transition_model(T=(0, 8), S=(0, 9), A='Right')

        self.assertEq(stay_from_corner, 0.90, "Incorrect transition model when staying in top right corner")

        for val in [down_from_corner, left_from_corner]:
            self.assertEq(val, 0.05, "Incorrect transition model when not staying in top right corner")

    def test_maximum_probable_explanation_edge(self) -> None:
        """
        Checks if maximum_probable_explanation is working properly.
        """
        actions = ['Left', 'Left']
        observations = [4, 4, 4]
        correct_states = [(4, 0), (4, 0), (4, 0)]
        assert maximum_probable_explanation(actions, observations) == correct_states, \
            "Correct state does not return [(4,0), (4,0), (4,0)]"

    def test_maximum_probable_explanation_corner(self) -> None:
        """
        Checks if maximum_probable_explanation is working properly.
        """
        actions = ['Down', 'Right']
        observations = [9, 9, 9]
        correct_states = [(9, 9), (9, 9), (9, 9)]
        assert maximum_probable_explanation(actions, observations) == correct_states, \
            "Correct state does not return [(9,9), (9,9), (9,9)]"

    def test_maximum_probable_explanation_nonreliable_actions(self) -> None:
        """
        Checks if maximum_probable_explanation is working properly.
        """
        # when the robot was told to go up, it didn't actually go up
        actions = ['Up', 'Left']
        observations = [9, 9, 9]
        correct_states = [(9, 9), (9, 9), (9, 8)]
        assert maximum_probable_explanation(actions, observations) == correct_states, \
            "Correct state does not return [(9,9), (9,9), (9,8)]"


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AdditionalUnitTests("test_cdf_basic"))
    suite.addTest(AdditionalUnitTests("test_cdf_2_value"))
    suite.addTest(AdditionalUnitTests("test_transition_model_edge"))
    suite.addTest(AdditionalUnitTests("test_transition_model_corner"))
    suite.addTest(AdditionalUnitTests("test_maximum_probable_explanation_edge"))
    suite.addTest(AdditionalUnitTests("test_maximum_probable_explanation_corner"))
    suite.addTest(AdditionalUnitTests("test_maximum_probable_explanation_nonreliable_actions"))
    runner = unittest.TextTestRunner()
    runner.run(suite)