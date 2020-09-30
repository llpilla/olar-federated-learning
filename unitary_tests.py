#!/usr/bin/env python3
"""Unitary tests for the different modules."""

import unittest
import numpy as np
import os

import code.devices as devices
import code.schedulers as schedulers
import code.support as support


class TestDevice(unittest.TestCase):
    def setUp(self):
        self.tests = 2
        self.size = 3
        self.matrix = np.zeros(shape=(self.tests, self.size+1))

    def test_linear(self):
        devices.create_linear_costs(0, self.matrix, 0, self.size)
        self.assertEqual(self.matrix[0][0], 5.939321535345923)
        self.assertEqual(self.matrix[0][1], 13.376025832697698)
        self.assertEqual(self.matrix[0][2], 20.812730130049474)
        self.assertEqual(self.matrix[0][3], 28.249434427401248)

        devices.create_linear_costs(10, self.matrix, 1, self.size)
        self.assertEqual(self.matrix[1][0], 7.941885789400714)
        self.assertEqual(self.matrix[1][1], 9.128653333635327)
        self.assertEqual(self.matrix[1][2], 10.31542087786994)
        self.assertEqual(self.matrix[1][3], 11.502188422104554)

    def test_quadratic(self):
        devices.create_quadratic_costs(20, self.matrix, 0, self.size)
        self.assertEqual(self.matrix[0][0], 6.293177209695468)
        self.assertEqual(self.matrix[0][1], 24.396377326152603)
        self.assertEqual(self.matrix[0][2], 60.54713057315448)
        self.assertEqual(self.matrix[0][3], 114.7454369507011)

        devices.create_quadratic_costs(30, self.matrix, 1, self.size)
        self.assertEqual(self.matrix[1][0], 6.797291824615015)
        self.assertEqual(self.matrix[1][1], 18.191459379352082)
        self.assertEqual(self.matrix[1][2], 43.52048923013118)
        self.assertEqual(self.matrix[1][3], 82.78438137695233)

    def test_nlogn(self):
        devices.create_nlogn_costs(40, self.matrix, 0, self.size)
        self.assertEqual(self.matrix[0][0], 4.669183252722576)
        self.assertEqual(self.matrix[0][1], 5.707721764703263)
        self.assertEqual(self.matrix[0][2], 7.961272446810852)
        self.assertEqual(self.matrix[0][3], 10.9004143246067)

        devices.create_nlogn_costs(50, self.matrix, 1, self.size)
        self.assertEqual(self.matrix[1][0], 5.451414809842193)
        self.assertEqual(self.matrix[1][1], 7.567418437443891)
        self.assertEqual(self.matrix[1][2], 12.158987612119443)
        self.assertEqual(self.matrix[1][3], 18.147436575452378)

    def test_recursive(self):
        devices.create_recursive_costs(60, self.matrix, 0, self.size)
        self.assertEqual(self.matrix[0][0], 3.7078599704195687)
        self.assertEqual(self.matrix[0][1], 6.390372317496471)
        self.assertEqual(self.matrix[0][2], 10.299016410210445)
        self.assertEqual(self.matrix[0][3], 17.290762542805226)

        devices.create_recursive_costs(70, self.matrix, 1, self.size)
        self.assertEqual(self.matrix[1][0], 9.34732484757063)
        self.assertEqual(self.matrix[1][1], 18.19916149725357)
        self.assertEqual(self.matrix[1][2], 24.460842950613547)
        self.assertEqual(self.matrix[1][3], 33.60737451412772)


class TestSchedulers(unittest.TestCase):
    def setUp(self):
        self.tasks = 4
        self.resources = 3
        self.lower_limit = np.zeros(shape=self.resources, dtype=int)
        self.upper_limit = np.full(shape=self.resources,
                                   fill_value=self.tasks+1)
        self.cost = np.array([[0.5, 0.5, 1.5, 2.0, 200],
                              [0.0, 0.4, 2.0, 3.0, 4.0],
                              [0.0, 1.5, 2.5, 3.5, 4.5]])

    def test_olar(self):
        assignment = schedulers.olar(self.tasks,
                                     self.resources,
                                     self.cost,
                                     self.lower_limit,
                                     self.upper_limit)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 1)

    def test_olar_l(self):
        lower_limit = np.array([0, 0, 3])
        assignment = schedulers.olar(self.tasks,
                                     self.resources,
                                     self.cost,
                                     lower_limit,
                                     self.upper_limit)
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 3)

    def test_olar_u(self):
        upper_limit = np.array([1, 3, 2])
        assignment = schedulers.olar(self.tasks,
                                     self.resources,
                                     self.cost,
                                     self.lower_limit,
                                     upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 1)

    def test_olar_l_u(self):
        lower_limit = np.array([0, 0, 2])
        upper_limit = np.array([1, 3, 2])
        assignment = schedulers.olar(self.tasks,
                                     self.resources,
                                     self.cost,
                                     lower_limit,
                                     upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 2)

    def test_fed_lbap(self):
        assignment = schedulers.fed_lbap(self.tasks,
                                         self.resources,
                                         self.cost)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 1)

    def test_fed_lbap_cornercase(self):
        tasks = 3
        resources = 2
        cost = np.array([[0.0, 1.0, 2.0, 2.5],
                         [0.0, 1.5, 2.0, 3.5]])
        assignment = schedulers.fed_lbap(tasks,
                                         resources,
                                         cost)
        # 4 tasks mapped instead of 3
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 2)

    def test_fed_lbap_cornercase2(self):
        tasks = 3
        resources = 2
        cost = np.array([[0.0, 1.0, 2.5, 2.5],
                         [0.0, 1.5, 2.0, 2.0]])
        assignment = schedulers.fed_lbap(tasks,
                                         resources,
                                         cost)
        # 4 tasks mapped instead of 3
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 3)

    def test_fedavg(self):
        tasks = 10
        resources = 3
        assignment = schedulers.fedavg(tasks, resources)
        self.assertEqual(assignment[0], 4)
        self.assertEqual(assignment[1], 3)
        self.assertEqual(assignment[2], 3)

    def test_random(self):
        tasks = 10
        resources = 3
        assignment = schedulers.random(tasks, resources, 0)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 5)
        self.assertEqual(assignment[2], 3)

        assignment = schedulers.random(tasks, resources, 10)
        self.assertEqual(assignment[0], 6)
        self.assertEqual(assignment[1], 0)
        self.assertEqual(assignment[2], 4)

    def test_proportional(self):
        assignment = schedulers.proportional(self.tasks,
                                             self.resources,
                                             self.cost,
                                             1)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 0)

        assignment = schedulers.proportional(self.tasks,
                                             self.resources,
                                             self.cost,
                                             2)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 1)

        assignment = schedulers.proportional(self.tasks,
                                             self.resources,
                                             self.cost,
                                             4)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 1)

    def test_extended_fed_lbap(self):
        assignment = schedulers.extended_fed_lbap(self.tasks,
                                                  self.resources,
                                                  self.cost,
                                                  self.lower_limit,
                                                  self.upper_limit)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 1)

    def test_extended_fed_lbap_l(self):
        lower_limit = np.array([0, 0, 3])
        assignment = schedulers.extended_fed_lbap(self.tasks,
                                                  self.resources,
                                                  self.cost,
                                                  lower_limit,
                                                  self.upper_limit)
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 3)

    def test_extended_fed_lbap_u(self):
        upper_limit = np.array([1, 3, 2])
        assignment = schedulers.extended_fed_lbap(self.tasks,
                                                  self.resources,
                                                  self.cost,
                                                  self.lower_limit,
                                                  upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 1)

    def test_extended_fed_lbap_l_u(self):
        # corner case fixed by the verification by the end
        lower_limit = np.array([0, 0, 2])
        upper_limit = np.array([1, 3, 2])
        assignment = schedulers.extended_fed_lbap(self.tasks,
                                                  self.resources,
                                                  self.cost,
                                                  lower_limit,
                                                  upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 2)

    def test_extended_fed_lbap_final_verification(self):
        tasks = 4
        resources = 3
        lower_limit = np.full(shape=self.resources,
                              fill_value=1)
        upper_limit = np.full(shape=self.resources,
                              fill_value=3)
        cost = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                         [0.0, 1.0, 2.0, 3.0, 4.2],
                         [0.0, 1.0, 2.0, 3.0, 4.5]])
        assignment = schedulers.extended_fed_lbap(tasks,
                                                  resources,
                                                  cost,
                                                  lower_limit,
                                                  upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 1)
        self.assertEqual(assignment[2], 2)

    def test_extended_fedavg(self):
        tasks = 10
        resources = 3
        lower_limit = np.full(shape=self.resources,
                              fill_value=1)
        upper_limit = np.full(shape=self.resources,
                              fill_value=5)
        assignment = schedulers.extended_fedavg(tasks, resources,
                                                lower_limit, upper_limit)
        self.assertEqual(assignment[0], 4)
        self.assertEqual(assignment[1], 3)
        self.assertEqual(assignment[2], 3)

        lower_limit[0] = 5
        upper_limit[2] = 2
        assignment = schedulers.extended_fedavg(tasks, resources,
                                                lower_limit, upper_limit)
        self.assertEqual(assignment[0], 5)
        self.assertEqual(assignment[1], 3)
        self.assertEqual(assignment[2], 2)

    def test_extended_proportional(self):
        assignment = schedulers.extended_proportional(
            self.tasks,
            self.resources,
            self.cost,
            1,
            self.lower_limit,
            self.upper_limit)
        self.assertEqual(assignment[0], 2)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 0)

        self.lower_limit[2] = 1
        self.upper_limit[0] = 1
        assignment = schedulers.extended_proportional(
            self.tasks,
            self.resources,
            self.cost,
            1,
            self.lower_limit,
            self.upper_limit)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(assignment[1], 2)
        self.assertEqual(assignment[2], 1)


class TestSupport(unittest.TestCase):
    def setUp(self):
        self.tasks = 4
        self.resources = 3
        self.cost = np.array([[0.5, 0.5, 1.5, 2.0, 200],
                              [0.0, 0.4, 2.0, 3.0, 4.0],
                              [0.0, 1.5, 2.5, 3.5, 4.5]])

    def test_makespan(self):
        assignment = np.array([0, 0, 0])
        makespan = support.get_makespan(self.cost, assignment)
        self.assertEqual(makespan, 0.5)

        assignment = np.array([4, 4, 4])
        makespan = support.get_makespan(self.cost, assignment)
        self.assertEqual(makespan, 200.0)

        assignment = np.array([2, 1, 1])
        makespan = support.get_makespan(self.cost, assignment)
        self.assertEqual(makespan, 1.5)

    def test_check_limits(self):
        assignment = np.array([4, 4, 4])
        lower_limit = np.array([1, 1, 1])
        upper_limit = np.array([5, 5, 5])
        check = support.check_limits(assignment, lower_limit, upper_limit)
        self.assertTrue(check)

        check = support.check_limits(assignment, assignment, assignment)
        self.assertTrue(check)

        assignment = np.array([0, 4, 4])
        check = support.check_limits(assignment, lower_limit, upper_limit)
        self.assertFalse(check)

        assignment = np.array([4, 4, 6])
        check = support.check_limits(assignment, lower_limit, upper_limit)
        self.assertFalse(check)

        assignment = np.array([0, 4, 6])
        check = support.check_limits(assignment, lower_limit, upper_limit)
        self.assertFalse(check)

    def test_check_total_assigned(self):
        assignment = np.array([4, 4, 4])
        tasks = 12
        check = support.check_total_assigned(tasks, assignment)
        self.assertTrue(check)

        tasks = 13
        check = support.check_total_assigned(tasks, assignment)
        self.assertFalse(check)

        tasks = 11
        check = support.check_total_assigned(tasks, assignment)
        self.assertFalse(check)


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.filename = 'dummy_file.txt'
        self.logger = support.Logger(self.filename)

    def test_attributes(self):
        self.assertFalse(self.logger.verbosity)
        self.assertEqual(self.filename, self.logger.filename)

    def test_logger(self):
        a = 'a'
        b = 'b'
        c = 'c'
        expected_log = 'ca\nb\n'
        self.logger.header(c)
        self.logger.store(a)
        self.logger.store(b)
        self.logger.finish()

        with open(self.filename, 'r') as logfile:
            written_log = logfile.read()
            self.assertEqual(expected_log, written_log)
        os.remove(self.filename)


if __name__ == '__main__':
    unittest.main()
