from Project2 import *
import unittest
from collections import Counter, Callable


class TestFunctions(unittest.TestCase):
    def test_functions_exist(self) -> None:
        """
        Checks if all the necessary functions exist.
        Autograder shall give an error at appropriate places if they don't.
        This will inform the user if some missing function caused the error.
        """
        assert 'action_prior' in globals(), "action_prior is missing"
        assert 'state_prior' in globals(), "state_prior is missing"
        assert 'calculate_cdf' in globals(), "calculate_cdf is missing"
        assert 'sample_from_pmf' in globals(), "sample_from_pmf is missing"
        assert 'sensor_model' in globals(), "sensor_model is missing"
        assert 'transition_model' in globals(), "transition_model is missing"
        assert 'maximum_probable_explanation' in globals(), "maximum_probable_explanation is missing"
        assert 'sample_from_sensor_model' in globals(), "sample_from_sensor_model is missing"
        assert 'sample_from_transition_model' in globals(), "sample_from_transition_model is missing"
        assert 'sample_from_dbn' in globals(), "sample_from_dbn is missing"

    def test_actions_exist(self) -> None:
        """
        Checks if ACTIONS variable exists.
        """
        assert 'ACTIONS' in globals(), "ACTIONS not found"
        self.assertEqual(len(ACTIONS), 4, "ACTIONS does not have four possibilities")

    def test_action_prior(self) -> None:
        """
        Checks if action_prior is working correctly.
        """
        prob_sum = 0
        for action in ACTIONS:
            p = action_prior(action)
            assert p is not None, "None returned by action_prior"
            self.assertGreaterEqual(p, 0, "Probability from action_prior cannot be lesser than 0")
            self.assertLessEqual(p, 1.0, "Probability from action_prior cannot be greater than 1")
            prob_sum += p
        # Takes care of floating-point errors in Python
        # because of how they are stored in memory
        self.assertGreaterEqual(prob_sum, 1.0 - 1e-6, "Sum of probabilities cannot be lesser than 1")
        self.assertLessEqual(prob_sum, 1.0 + 1e-6, "Sum of probabilities cannot be greater than 1")


        for action in ACTIONS:
            p = action_prior(action)
            if action == 'Left': self.assertEquals(p, 0.2, '\"Left\" value in action_prior pmf is incorrect.')
            if action == 'Right': self.assertEquals(p, 0.6, '\"Right\" value in action_prior pmf is incorrect.')
            if action == 'Up': self.assertEquals(p, 0.1, '\"Up\" value in action_prior pmf is incorrect.')
            if action == 'Down': self.assertEquals(p, 0.1, '\"Down\" value in action_prior pmf is incorrect.')


    def test_states_exist(self) -> None:
        """
        Checks if STATES variable exists.
        """
        assert 'STATES' in globals(), "STATES not found"
        self.assertEqual(len(STATES), 100, "STATES does not have 100 possibilities")

    def test_state_prior(self) -> None:
        """
        Checks if state_prior is working correctly.
        """
        prob_sum = 0
        for state in STATES:
            p = state_prior(state)
            assert p is not None, "None returned by state_prior"
            self.assertGreaterEqual(p, 0, "Probability from state_prior cannot be lesser than 0")
            self.assertLessEqual(p, 1.0, "Probability from state_prior cannot be greater than 1")
            prob_sum += p
        # Takes care of floating-point errors in Python
        # because of how they are stored in memory
        self.assertGreaterEqual(prob_sum, 1.0 - 1e-6, "Sum of probabilities cannot be lesser than 1")
        self.assertLessEqual(prob_sum, 1.0 + 1e-6, "Sum of probabilities cannot be greater than 1")

        for state in STATES:
            p = state_prior(state)
            if   state == (4,0): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (1,1): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (7,2): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (3,3): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (0,4): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (4,5): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (7,6): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (8,7): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (4,8): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            elif state == (9,9): self.assertEquals(p, 0.1, "Wrong values for state " + str(state) + " in state_prior pmf.")
            else: self.assertEquals(p, 0, "Wrong values for state " + str(state) + " in state_prior pmf.")

    def test_calculate_cdf(self) -> None:
        """
        Checks if CDF is working properly.
        """
        cdf = calculate_cdf(action_prior, ACTIONS)
        assert callable(cdf), "calculate_cdf must return a function"
        curr_sum = 0
        assert curr_sum < cdf("Left"), "CDF must go in ascending order for the actions passed in"
        curr_sum = cdf("Left")
        assert 0.2 - 1e-6 < curr_sum < 0.2 + 1e-6, f"Final CDF Value must 0.2, {curr_sum} found"

    def pmf_action_sanity_check(self) -> None:
        """
        Checks if pmf returns an action for actions.
        """
        sample_num_10 = [sample_from_pmf(action_prior, ACTIONS)]
        for curr_sample in sample_num_10:
            assert curr_sample in ACTIONS, "PMF returns something other than allowable outputs for actions"

    def test_sensor_model(self) -> None:
        """
        Checks if the sensor model is working properly.
        """
        curr_state = (3, 6)
        for i in range(0, 3):
            assert sensor_model(i, curr_state) == 0.01, \
                "Sensor model does not return 0.01 for incorrect observation."
        assert sensor_model(3, curr_state) == 0.91, \
            "Sensor model does not return 0.01 for incorrect observation."
        for i in range(4, 10):
            assert sensor_model(i, curr_state) == 0.01, \
                "Sensor model does not return 0.01 for incorrect observation."

        for state in STATES:
            for z in range(10):
                p = sensor_model(z, state)
                if state[0] == z:
                    self.assertEquals(p, 0.91, "Wrong value for state " + str(state) + " and observation (" + str(z) + \
                                               ") in sensor_model pmf.")
                else:
                    self.assertEquals(p, 0.01, "Wrong value for state " + str(state) + " and observation (" + str(z) + \
                                               ") in sensor_model pmf.")

    def test_transition_model_center_normal(self) -> None:
        """
        Checks if the transition model is working properly if the robot is neither on the edge nor in the corner.
        """
        curr_state = (3, 6)
        curr_action = "Right"
        correct_state = (3, 7)
        incorrect_states = [(3, 5), (2, 6), (4, 6)]
        invalid_states = [(0, 0), (5, 5), (6, 3)]
        assert transition_model(correct_state, curr_state, curr_action) == 0.85, \
            "Correct state does not return 0.85"
        for curr_incorrect_state in incorrect_states:
            assert transition_model(curr_incorrect_state, curr_state, curr_action) == 0.05, \
                "Incorrect state does not return 0.05"
        for curr_invalid_state in invalid_states:
            assert transition_model(curr_invalid_state, curr_state, curr_action) == 0, \
                "Invalid state does not return 0"

    def test_transition_model_corner_wall(self) -> None:
        """
        Checks if the transition model is working properly if the robot is on the edge and is moving towards wall.
        """
        curr_state = (0, 0)
        curr_action = "Left"
        correct_state = (0, 0)
        incorrect_states = [(0, 1), (1, 0)]
        invalid_states = [(1, 1), (5, 5), (6, 3)]
        assert transition_model(correct_state, curr_state, curr_action) == 0.90, \
            "Correct state does not return 0.90"
        for curr_incorrect_state in incorrect_states:
            assert transition_model(curr_incorrect_state, curr_state, curr_action) == 0.05, \
                "Incorrect state does not return 0.05"
        for curr_invalid_state in invalid_states:
            assert transition_model(curr_invalid_state, curr_state, curr_action) == 0, \
                "Invalid state does not return 0"

    def test_comprehensive_transition_model(self) -> None:
        """
        Checks all states and transitions (takes some time to run)
        """
        for start in STATES:
            for end in STATES:
                for a in ACTIONS:
                    p = transition_model(end, start, a)
                    if 9 > start[0] > 0 and 9 > start[1] > 0:
                        if   end[1] == start[1] and end[0] == start[0] - 1 and a != 'Up': self.assertEquals(p, 0.05, "Wrong value for transition_model"\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] + 1 and a != 'Down': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] - 1 and a != 'Left': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] + 1 and a != 'Right': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] - 1 and a == 'Up': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] + 1 and a == 'Down': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] - 1 and a == 'Left': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] + 1 and a == 'Right': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] in [0, 9] and start[0] in [0, 9]:
                        if start == (0,0):
                            if a == 'Up':
                                if   end == (0,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (0,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (0,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (0,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (0,9):
                            if a == 'Up':
                                if   end == (0,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (0,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (0,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (0,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (9,0):
                            if a == 'Up':
                                if   end == (9,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (9,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (9,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (9,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (9,9):
                            if a == 'Up':
                                if   end == (9,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (9,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (9,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (9,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] == 0:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 0 and end[1] == start[1] - 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 0 and end[1] == start[1] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 1 and end[1] == start[1]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] == 9:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 9 and end[1] == start[1] - 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 9 and end[1] == start[1] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 8 and end[1] == start[1]:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[1] == 0:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 0 and end[0] == start[0] - 1:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 0 and end[0] == start[0] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 1 and end[0] == start[0]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[1] == 9:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 9 and end[0] == start[0] - 1:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 9 and end[0] == start[0] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 8 and end[0] == start[0]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')

    def test_comprehensive_portal_transition_model_extra_credit(self) -> None:
        """
        Checks all states and transitions for the portal extra credit model (takes some time to run)
        """
        for start in STATES:
            for end in STATES:
                for a in ACTIONS:
                    p = transition_model_portal(end, start, a)
                    if start == (3,4) and (end == (6,3) or end == (3,5)):
                        if end == (6,3):
                            if a == "Right": self.assertEquals(p, 0.85, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            else: self.assertEquals(p, 0.05, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end == (3,5): self.assertEquals(p, 0, "Wrong value for ptransition_model"\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start == (0,1) and (end == (5,5) or end == (0,1)):
                        if end == (5,5):
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            else: self.assertEquals(p, 0.05, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end == (0,1): self.assertEquals(p, 0, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start == (3,8) and (end == (0,0) or end == (3,7)):
                        if end == (0,0):
                            if a == "Left": self.assertEquals(p, 0.85, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            else: self.assertEquals(p, 0.05, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end == (3,7): self.assertEquals(p, 0, "Wrong value for ptransition_model"\
                                                            + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif 9 > start[0] > 0 and 9 > start[1] > 0:
                        if   end[1] == start[1] and end[0] == start[0] - 1 and a != 'Up': self.assertEquals(p, 0.05, "Wrong value for ptransition_model"\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] + 1 and a != 'Down': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] - 1 and a != 'Left': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] + 1 and a != 'Right': self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] - 1 and a == 'Up': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == start[1] and end[0] == start[0] + 1 and a == 'Down': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] - 1 and a == 'Left': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == start[0] and end[1] == start[1] + 1 and a == 'Right': self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                                + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] in [0, 9] and start[0] in [0, 9]:
                        if start == (0,0):
                            if a == 'Up':
                                if   end == (0,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (0,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (0,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (0,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,1): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (0,9):
                            if a == 'Up':
                                if   end == (0,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (0,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (0,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (0,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (0,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (1,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (9,0):
                            if a == 'Up':
                                if   end == (9,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (9,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (9,0): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (9,0): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,0): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,1): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif start == (9,9):
                            if a == 'Up':
                                if   end == (9,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Down':
                                if   end == (9,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Left':
                                if   end == (9,9): self.assertEquals(p, 0.10, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == 'Right':
                                if   end == (9,9): self.assertEquals(p, 0.90, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (9,8): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                elif end == (8,9): self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                                else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] == 0:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 0 and end[1] == start[1] - 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 0 and end[1] == start[1] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 1 and end[1] == start[1]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[0] == 9:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 9 and end[1] == start[1] - 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 9 and end[1] == start[1] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[0] == 8 and end[1] == start[1]:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[1] == 0:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 0 and end[0] == start[0] - 1:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 0 and end[0] == start[0] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 1 and end[0] == start[0]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                    elif start[1] == 9:
                        if end == start:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 9 and end[0] == start[0] - 1:
                            if a == "Up": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 9 and end[0] == start[0] + 1:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        elif end[1] == 8 and end[0] == start[0]:
                            if a == "Up": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Down": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Left": self.assertEquals(p, 0.85, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                            elif a == "Right": self.assertEquals(p, 0.05, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')
                        else: self.assertEquals(p, 0, "Wrong value for transition_model "\
                                                 + "(T, S, A) => (" + str(end) + ', ' + str(start) + ', ' + str(a) + ').')

    def test_calculate_cdf(self) -> None:

        def pmf(val):
            return {'0.1': 0.1, '0.5': 0.4, '0.85': 0.35, '1.0': 0.15}[val]

        keys = ['0.1', '0.5', '0.85', '1.0']
        cdf = calculate_cdf(pmf, keys)

        self.assertIsInstance(cdf, Callable, "calculate_cdf did not return a function.")

        for key in keys:
            self.assertEquals(cdf(key), float(key), 'Wrong value for calculate_cdf.')

    def test_sample_sensor_model(self) -> None:
        """
        Checks if sample_from_sensor_model is working properly.
        """
        correct_state = (1, 1)
        samples = [sample_from_sensor_model(correct_state) for _ in range(10000)]
        hist = np.array([Counter(samples)[obs] for obs in OBSERVATIONS])
        hist = hist / np.sum(hist)  # normalize
        sensor_model_flattened = np.array([0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        np.testing.assert_allclose(hist, sensor_model_flattened, atol=1e-2, rtol=1e-2)

    def test_sample_transition_model(self) -> None:
        """
        Checks if sample_from_transition_model is working properly.
        """
        curr_state = (1, 1)
        action = "Right"
        samples = [sample_from_transition_model(curr_state, action) for _ in range(10000)]
        hist = np.array([Counter(samples)[act] for act in [(1, 2), (0, 1), (2, 1), (1, 0)]])
        hist = hist / np.sum(hist)  # normalize
        transition_model_flattened = np.array([0.85, 0.05, 0.05, 0.05])
        np.testing.assert_allclose(hist, transition_model_flattened, atol=1e-1, rtol=1e-1)

    def test_sample_from_dbn_types(self) -> None:
        """
        Checks if sample_from_dbn returns the correct types.
        """
        samples = sample_from_dbn()
        for state in samples['states']:
            assert state in STATES
        for obs in samples['observations']:
            assert obs in OBSERVATIONS
        for action in samples['actions']:
            assert action in ACTIONS

    def test_maximum_probable_explanation(self) -> None:
        """
        Checks if maximum_probable_explanation is working properly.
        """
        actions = ['Down', 'Down']
        observations = [3, 4, 5]
        correct_states = [(3, 3), (4, 3), (5, 3)]
        assert maximum_probable_explanation(actions, observations) == correct_states, \
            "Correct state does not return [(3,3), (3,4), (3,5)]"

    def ape_test_sample_pmf(self) -> None:
        def pmf(item):
            if (item == "one"):
                return 0.25
            elif (item == "two"):
                return 0.15
            elif (item == "three"):
                return 0.1
            else:
                return 0.5
      
        results = {"one": 0,"two": 0,"three": 0,"four": 0}
        orderedList = ["one", "two", "three", "four"]
        for i in range(10000):
            state = sample_from_pmf(pmf, orderedList)
            results[state] += 1
        print("\nSample From PMF Test")
        print("\nExpected Distribution: {'one': 2500, 'two': 2500, 'three': 1000, 'four': 5000}")
        print("Your Distribution:", end =" ")
        print(results)

    def ape_test_sample_sensor(self) -> None:
        results = {0: 0,1: 0,2: 0,3: 0,4: 0,5: 0,6: 0,7: 0,8: 0,9: 0}
        state = (1,1)
        for i in range(10000):
            answer = sample_from_sensor_model(state)
            results[answer] += 1
        print("\nSample From Sensor Test")
        print("\nExpected Distribution: {0: 100, 1: 9100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 100, 9: 100}")
        print("Your Distribution:", end =" ")
        print(results)


    def ape_test_sample_transition(self) -> None:
        results = {}
        for i in range(10):
            for j in range(10):
                results[(i,j)] = 0
        
        state = (1,1)
        action = "Up"
        for i in range(10000):
            answer = sample_from_transition_model(state, action)
            results[answer] += 1
        print("\nSample From Transition Test")
        print("Expected Distribution: {(0,1): 8500, (1,2): 500, (1,0): 500, (2,1): 500}")
        print("Your Distribution:", end =" ")
        print(results)

        for i in range(10):
            for j in range(10):
                results[(i,j)] = 0
        state = (0,0)
        action = "Left"
        for i in range(10000):
            answer = sample_from_transition_model(state, action)
            results[answer] += 1
        print("\nSample From Transition Test [(0,0) edge case]")
        print("Expected Distribution: {(0,0): 9000, (1,0): 500, (0,1): 500}")
        print("Your Distribution:", end =" ")
        print(results)


def suite():
    functions = [
        'test_functions_exist',
        'test_actions_exist',
        'test_action_prior',
        'test_states_exist',
        'test_state_prior',
        'test_calculate_cdf',
        'pmf_action_sanity_check',
        'test_sensor_model',
        'test_transition_model_center_normal',
        'test_transition_model_corner_wall',
        'test_sample_from_dbn_types',
        'test_maximum_probable_explanation',
        'test_sample_sensor_model',
        'test_sample_transition_model',
        'test_comprehensive_transition_model',
        'test_comprehensive_portal_transition_model_extra_credit',
        'test_calculate_cdf',

        # ape tests
        # 'ape_test_sample_pmf',
        # 'ape_test_sample_sensor',
        # 'ape_test_sample_transition'
    ]
    suite = unittest.TestSuite()
    for func in functions:
        suite.addTest(TestFunctions(func))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())